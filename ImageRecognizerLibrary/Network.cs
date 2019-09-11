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

        public event Action<UIImage> ShowedImage;
        public event Action<UIImage> ShowedOutputImage;

        protected static readonly IMPSNNPadding sameConvPadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToBottomLeft | MPSNNPaddingMethod.AlignCentered);
        protected static readonly IMPSNNPadding validConvPadding = MPSNNDefaultPadding.Create (
            MPSNNPaddingMethod.AddRemainderToBottomLeft | MPSNNPaddingMethod.AlignCentered | MPSNNPaddingMethod.SizeValidOnly);
        protected static readonly IMPSNNPadding samePoolingPadding = MPSNNDefaultPadding.CreatePaddingForTensorflowAveragePooling ();
        protected static readonly IMPSNNPadding validPoolingPadding = MPSNNDefaultPadding.CreatePaddingForTensorflowAveragePoolingValidOnly ();

        protected Network ()
        {
            device = MTLDevice.SystemDefault;
            if (device == null)
                throw new InvalidOperationException ("No Metal Devices");
            Console.WriteLine ($"Device: {device.Name}");

            var queue = device.CreateCommandQueue ();

            Console.WriteLine ($"Queue: {queue}");
            Console.WriteLine ($"Queue.Label: {queue.Label}");

            compileOptions = new CompileOptions (new ExecutionOptions (device, queue));

            textureLoader = new MTKTextureLoader (device);
        }

        public override string ToString ()
        {
            return $"Network ({weights.Count} weights)";
        }

        public async Task TrainAsync ()
        {
            await TrainBatchesAsync ().ConfigureAwait (false);
            foreach (var g in Graphs) {
                g.ReloadFromDataSources ();
            }
        }

        public Task PredictAsync ()
        {
            return PredictBatchesAsync ();
        }

        protected abstract Task TrainBatchesAsync ();
        protected abstract Task PredictBatchesAsync ();

        protected Conv2dWeights Conv2d (int inChannels, int outChannels, string label, int kernelSize = 3, int stride = 1, bool bias = true)
        {
            if (weightsIndex.TryGetValue (label, out var w) && w is Conv2dWeights cw)
                return cw;
            Console.WriteLine ($"Create weights {label}");
            w = cw = new Conv2dWeights (compileOptions, inChannels, outChannels, kernelSize: kernelSize, stride: stride, bias: bias, label: label, seed: 42+weights.Count);

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

        protected async Task<NSArray<MPSImage>> LoadImagesAsync (params object[] imageSources)
        {
            var images = await Task.WhenAll (imageSources.Select (x => LoadImageAsync (x))).ConfigureAwait (false);
            return NSArray<MPSImage>.FromNSObjects (images);
        }

        protected async Task<MPSImage> LoadImageAsync (object imageSource)
        {
            var textureOptions = new MTKTextureLoaderOptions {
                AllocateMipmaps = false,
                Srgb = true,
                TextureUsage = MTLTextureUsage.ShaderRead,
            };
            switch (imageSource) {
                case NSUrl url: {
                        //Console.WriteLine ($"Loading {Path.GetFileName(url.Path)}");
                        var texture = await textureLoader.FromUrlAsync (url, textureOptions);
                        var image = new MPSImage (texture, 3);
                        //Console.WriteLine ($"Done loading Image (type={image.ImageType}) from {Path.GetFileName (url.Path)}");
                        ShowImage (image);
                        return image;
                    }
                default:
                    throw new NotSupportedException ($"Cannot use {imageSource} ({imageSource.GetType ().Name}) as image source");
            }
        }

        protected unsafe UIKit.UIImage GetUIImage (MPSImage mpsImage)
        {
            var width = (int)mpsImage.Width;
            var height = (int)mpsImage.Height;
            var nfc = (int)mpsImage.FeatureChannels;
            var obytesPerRow = 4 * width;
            var cellSize = 44;
            using var cs = CoreGraphics.CGColorSpace.CreateDeviceRGB ();
            //Console.WriteLine ((width, height, mpsImage.Precision, mpsImage.PixelSize, mpsImage.FeatureChannels, mpsImage.PixelFormat, mpsImage.FeatureChannelFormat));
            if (mpsImage.FeatureChannelFormat == MPSImageFeatureChannelFormat.Float32 && nfc == 3) {
                var data = new float[width * height * nfc];
                fixed (float* dataPointer = data) {
                    mpsImage.ReadBytes ((IntPtr)dataPointer, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                }
                using var bc = new CoreGraphics.CGBitmapContext (null, width, height, 8, obytesPerRow, cs, CoreGraphics.CGImageAlphaInfo.NoneSkipFirst);
                var pixels = (byte*)bc.Data;
                var p = pixels;
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        *p++ = 255;
                        *p++ = ClampRGBA32Float (data[y * (width * 3) + x * 3 + 2] / 8);
                        *p++ = ClampRGBA32Float (data[y * (width * 3) + x * 3 + 1] / 8);
                        *p++ = ClampRGBA32Float (data[y * (width * 3) + x * 3 + 0] / 8);
                    }
                }
                var cgimage = bc.ToImage ();
                //Console.WriteLine ($"pixels f32 = " + string.Join (", ", data.Skip (data.Length / 2).Take (12)));
                return UIImage.FromImage (cgimage);
            }
            else if (mpsImage.FeatureChannelFormat == MPSImageFeatureChannelFormat.Float32 && nfc == 1) {
                var data = new float[width * height * nfc];
                fixed (float* dataPointer = data) {
                    mpsImage.ReadBytes ((IntPtr)dataPointer, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                }
                using var bc = new CoreGraphics.CGBitmapContext (null, width, height, 8, obytesPerRow, cs, CoreGraphics.CGImageAlphaInfo.NoneSkipFirst);
                var pixels = (byte*)bc.Data;
                var p = pixels;
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        var g = ClampRGBA32Float (data[y * width + x]);
                        *p++ = 255;
                        *p++ = g;
                        *p++ = g;
                        *p++ = g;
                    }
                }
                var cgimage = bc.ToImage ();
                //Console.WriteLine ($"pixels f32 = " + string.Join (", ", data.Skip (data.Length / 2).Take (12)));
                return UIImage.FromImage (cgimage);
            }
            else if (mpsImage.FeatureChannelFormat == MPSImageFeatureChannelFormat.Unorm8 && nfc == 3) {
                var data = new byte[width * height * (int)mpsImage.FeatureChannels];
                fixed (byte* dataPointer = data) {
                    mpsImage.ReadBytes ((IntPtr)dataPointer, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                    //mpsImage.Texture.GetBytes ((IntPtr)dataPointer, (nuint)(4 * width), MTLRegion.Create3D (0, 0, 0, width, height, 1), 0);
                }
                using var bc = new CoreGraphics.CGBitmapContext (null, width, height, 8, obytesPerRow, cs, CoreGraphics.CGImageAlphaInfo.NoneSkipFirst);
                var pixels = (byte*)bc.Data;
                var p = pixels;
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        *p++ = 255;
                        *p++ = data[y * (width * 3) + x * 3 + 2]; // Red
                        *p++ = data[y * (width * 3) + x * 3 + 1]; // Green
                        *p++ = data[y * (width * 3) + x * 3 + 0]; // Blue
                    }
                }
                var cgimage = bc.ToImage ();
                //Console.WriteLine ($"pixels 3 unorm8 = " + string.Join (", ", data.Skip (data.Length / 2).Take (12)));
                return UIImage.FromImage (cgimage);
            }
            else if (mpsImage.FeatureChannelFormat == MPSImageFeatureChannelFormat.Unorm8 && nfc == 1) {
                var data = new byte[width * height * (int)mpsImage.FeatureChannels];
                fixed (byte* dataPointer = data) {
                    mpsImage.ReadBytes ((IntPtr)dataPointer, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                    //mpsImage.Texture.GetBytes ((IntPtr)dataPointer, (nuint)(4 * width), MTLRegion.Create3D (0, 0, 0, width, height, 1), 0);
                }
                using var bc = new CoreGraphics.CGBitmapContext (null, width, height, 8, obytesPerRow, cs, CoreGraphics.CGImageAlphaInfo.NoneSkipFirst);
                var pixels = (byte*)bc.Data;
                var p = pixels;
                for (var y = 0; y < height; y++) {
                    for (var x = 0; x < width; x++) {
                        var g = data[y * width + x]; // Red
                        *p++ = 255;
                        *p++ = g;
                        *p++ = g;
                        *p++ = g;
                    }
                }
                var cgimage = bc.ToImage ();
                //Console.WriteLine ($"pixels 1 unorm8 = " + string.Join (", ", data.Skip (data.Length / 2).Take (12)));
                return UIImage.FromImage (cgimage);
            }
            else if (mpsImage.FeatureChannelFormat == MPSImageFeatureChannelFormat.Float32 && width == 1 && height == 1) {
                var data = new float[width * height * nfc];
                fixed (void* dataPointer = data) {
                    mpsImage.ReadBytes ((IntPtr)dataPointer, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                }
                return DrawCells (nfc, cellSize, data);
            }
            else if (mpsImage.FeatureChannelFormat == MPSImageFeatureChannelFormat.Unorm8 && width == 1 && height == 1) {
                var data = new byte[width * height * nfc];
                fixed (void* dataPointer = data) {
                    mpsImage.ReadBytes ((IntPtr)dataPointer, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                }
                return DrawCells (nfc, cellSize, data.Select (x => x / 255.0f).ToArray ());
            }
            else {
                if (width == 1 && height == 1) {
                    width = cellSize;
                    height = cellSize;
                }
                UIGraphics.BeginImageContext (new CoreGraphics.CGSize (width, height));
                UIColor.Red.SetColor ();
                var m = $"{mpsImage.FeatureChannels}{mpsImage.FeatureChannelFormat}?";
                m.DrawString (new CoreGraphics.CGPoint (0, 0), UIFont.SystemFontOfSize (8));
                var image = UIGraphics.GetImageFromCurrentImageContext ();
                UIGraphics.EndImageContext ();
                return image;
            }
        }

        static unsafe UIImage DrawCells (int nfc, int cellSize, float[] data)
        {
            var width = cellSize * nfc;
            var height = cellSize * nfc;
            UIGraphics.BeginImageContext (new CoreGraphics.CGSize (width, height));
            var maxi = Array.IndexOf (data, data.Max ());
            for (var i = 0; i < nfc; i++) {
                var v = Math.Clamp (data[i], -1f, 1f);
                var r = v < 0 ? 255 : 0;
                var g = v >= 0 ? 255 : 0;
                UIColor.FromRGBA (r, g, 0, (byte)(255 * Math.Abs (v))).SetColor ();
                UIGraphics.RectFill (new CoreGraphics.CGRect (i * cellSize, 0, cellSize, cellSize));
                if (i == maxi) {
                    i.ToString ().DrawString (new CoreGraphics.CGPoint (i * cellSize, cellSize), UIFont.SystemFontOfSize (cellSize));
                }
                UIColor.FromWhiteAlpha (1.0f, 0.5f).SetColor ();
                UIGraphics.RectFrame (new CoreGraphics.CGRect (i * cellSize, 0, cellSize, cellSize));
            }
            var image = UIGraphics.GetImageFromCurrentImageContext ();
            UIGraphics.EndImageContext ();
            return image;
        }

        static byte ClampRGBA32Float (float v)
        {
            if (v <= 0.0f)
                return 0;
            if (v >= 255.0f)
                return 255;
            return (byte)(v);
        }

        protected void ShowImage (MPSImage image)
        {
            ShowImage (GetUIImage (image));
        }

        protected void ShowOutputImage (MPSImage image)
        {
            ShowedOutputImage?.Invoke (GetUIImage (image));
        }

        protected void ShowOutputImage (UIImage image)
        {
            ShowedOutputImage?.Invoke (image);
        }

        protected void ShowImage (UIImage image)
        {
            ShowedImage?.Invoke (image);
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
