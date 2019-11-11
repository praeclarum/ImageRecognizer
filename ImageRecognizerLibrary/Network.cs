#nullable enable

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalKit;
using MetalPerformanceShaders;
using UIKit;
using System.IO.Compression;
using Google.Protobuf;
using static ImageRecognizerLibrary.MPSExtensions;

namespace ImageRecognizerLibrary
{
    public abstract class Network
    {
        protected readonly IMTLDevice device;

        readonly MTKTextureLoader textureLoader;

        readonly List<IWeights> weights = new List<IWeights> ();
        readonly Dictionary<string, IWeights> weightsIndex = new Dictionary<string, IWeights> ();

        protected readonly CompileOptions compileOptions;

        protected const MPSImageFeatureChannelFormat FeatureChannelFormat = MPSImageFeatureChannelFormat.Float32;

        protected static readonly IMPSNNPadding sameConvPadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToBottomLeft | MPSNNPaddingMethod.AlignCentered);
        protected static readonly IMPSNNPadding validConvPadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToBottomLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeValidOnly);
        protected static readonly IMPSNNPadding samePoolingPadding = MPSNNDefaultPadding.CreatePaddingForTensorflowAveragePooling ();
        protected static readonly IMPSNNPadding validPoolingPadding = MPSNNDefaultPadding.CreatePaddingForTensorflowAveragePoolingValidOnly ();

        public event Action<(UIImage, UIImage?)> ImagesShown;

        protected Network ()
        {
            device = MTLDevice.SystemDefault;
            if (device == null)
                throw new InvalidOperationException ("No Metal Devices");
            Console.WriteLine ($"Device: {device.Name}");

            compileOptions = new CompileOptions (new ExecutionOptions (device));

            textureLoader = new MTKTextureLoader (device);
        }

        public override string ToString ()
        {
            return $"Network ({weights.Count} weights)";
        }

        public async Task TrainAsync (IDataSet dataSet)
        {
            await TrainBatchesAsync (dataSet).ConfigureAwait (false);
            foreach (var g in Graphs) {
                g.ReloadFromDataSources ();
            }
        }

        public Task PredictAsync (IDataSet dataSet)
        {
            return PredictBatchesAsync (dataSet);
        }

        protected abstract Task TrainBatchesAsync (IDataSet dataSet);
        protected abstract Task PredictBatchesAsync (IDataSet dataSet);

        protected ConvolutionWeights Conv2d (int inChannels, int outChannels, string label, int kernelSize = 3, int stride = 1, bool bias = true)
        {
            if (weightsIndex.TryGetValue (label, out var w) && w is ConvolutionWeights cw)
                return cw;

            Console.WriteLine ($"Create weights {label}");
            w = cw = new ConvolutionWeights (compileOptions, inChannels, outChannels, kernelSize: kernelSize, stride: stride, bias: bias, label: label, seed: 42+weights.Count);

            weights.Add (w);
            weightsIndex[label] = w;
            return cw;
        }

        protected void DumpWeights ()
        {
            foreach (var wss in weights) {
                var ws = wss.GetWeights ();
                foreach (var w in ws) {
                    Console.WriteLine ($"{w.Key} = [" + string.Join (", ", w.Value.Take (5)) + "]");
                }
            }
        }

        public bool WeightsAreValid ()
        {
            return weights.All (x => x.WeightsAreValid ());
        }

        protected void ShowImages (MPSImage image, MPSImage outputImage)
        {
            ImagesShown?.Invoke ((ImageConversion.GetUIImage (image), ImageConversion.GetUIImage (outputImage)));
        }

        public void Write (string path, bool includeOptimizationParameters = false)
        {
            using (var stream = new FileStream (path, FileMode.Create, FileAccess.Write)) {
                Write (stream, includeOptimizationParameters: includeOptimizationParameters);
            }
            var length = new FileInfo (path).Length;
            Console.WriteLine ($"Wrote {length:#,0} bytes to {path}");
        }

        public void Write (Stream stream, bool includeOptimizationParameters = false)
        {
            using var gz = new GZipStream (stream, CompressionLevel.Fastest);

            var data = new NetworkData.Network ();
            foreach (var w in weights) {
                data.DataSources[w.Name] = w.GetData (includeTrainingParameters: includeOptimizationParameters);
            }

            data.WriteTo (gz);
        }

        public Task WriteAsync (string path, bool includeOptimizationParameters = false) =>
            Task.Run (() => Write (path, includeOptimizationParameters: includeOptimizationParameters));

        public static NetworkData.Network ReadData (string path)
        {
            //DumpWeights ();
            using var stream = new FileStream (path, FileMode.Open, FileAccess.Read);
            using var gz = new GZipStream (stream, CompressionMode.Decompress);

            var data = new NetworkData.Network ();
            data.MergeFrom (gz);
            return data;
        }

        public static Task<NetworkData.Network> ReadDataAsync (string path) => Task.Run (() => ReadData (path));

        public async Task ReadAsync (string path)
        {
            var data = await ReadDataAsync (path).ConfigureAwait (false);
            foreach (var d in data.DataSources) {
                if (weightsIndex.TryGetValue (d.Key, out var w)) {
                    w.SetData (d.Value);
                }
                foreach (var g in Graphs) {
                    g.ReloadFromDataSources ();
                }
            }
            Console.WriteLine ($"Read {data.DataSources.Count} weights from {path}");
        }

        public abstract MPSNNGraph[] Graphs { get; }
    }

    public class ExecutionOptions
    {
        public readonly IMTLDevice Device;
        public readonly IMTLCommandQueue Queue;

        public ExecutionOptions (IMTLDevice device, IMTLCommandQueue queue)
        {
            Device = device;
            Queue = queue;
        }

        public ExecutionOptions (IMTLDevice device)
        {
            Device = device;
            Queue = device.CreateCommandQueue ();
            Console.WriteLine ($"Queue: {Queue}");
            Console.WriteLine ($"Queue.Label: {Queue.Label}");
        }
    }

    public class CompileOptions
    {
        public readonly ExecutionOptions ExecutionOptions;
        public IMTLDevice Device => ExecutionOptions.Device;
        public IMTLCommandQueue Queue => ExecutionOptions.Queue;

        public float LearningRate = 1.0e-3f;

        public CompileOptions (ExecutionOptions executionOptions)
        {
            ExecutionOptions = executionOptions;
        }
    }
}
