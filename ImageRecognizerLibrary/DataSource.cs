#nullable enable

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using Foundation;
using Metal;
using MetalPerformanceShaders;

using static ImageRecognizerLibrary.MPSExtensions;

namespace ImageRecognizerLibrary
{
    public interface IWeights
    {
        Dictionary<string, float[]> GetWeights ();
        string Name { get; }
        NetworkData.DataSource GetData (bool includeTrainingParameters);
        void SetData (NetworkData.DataSource data);
        bool WeightsAreValid ();
    }

    public class Conv2dWeights : MPSCnnConvolutionDataSource, IWeights
    {
        readonly string label;
        readonly bool bias;
        readonly MPSCnnConvolutionDescriptor descriptor;

        nuint updateCount;
        readonly MPSNNOptimizerAdam updater;

        readonly OptimizerVectors weightVectors;
        readonly OptimizerVectors biasVectors;
        MPSCnnConvolutionWeightsAndBiasesState convWtsAndBias;
        readonly NSArray<MPSVector> momentumVectors;
        readonly NSArray<MPSVector> velocityVectors;

        //bool isLoaded = false;
        bool isDisposed = false;

        public override string Label => label;

        public string Name => label;

        public override MPSCnnConvolutionDescriptor Descriptor => descriptor;

        public override MPSDataType DataType => MPSDataType.Float32;

        public override IntPtr Weights => weightVectors.ValuePointer;

        public override IntPtr BiasTerms => biasVectors.ValuePointer;

        readonly ExecutionOptions options;

        public Conv2dWeights (CompileOptions options, int inChannels, int outChannels, int kernelSize, int stride, bool bias, string label, int seed)
        {
            this.options = options.ExecutionOptions;

            descriptor = MPSCnnConvolutionDescriptor.CreateCnnConvolutionDescriptor (
                (System.nuint)kernelSize, (System.nuint)kernelSize,
                (System.nuint)inChannels,
                (System.nuint)outChannels);
            descriptor.StrideInPixelsX = (nuint)stride;
            descriptor.StrideInPixelsY = (nuint)stride;
            this.bias = bias;
            this.label = string.IsNullOrEmpty (label) ? Guid.NewGuid ().ToString () : label;

            var lenWeights = inChannels * kernelSize * kernelSize * outChannels;

            var vDescWeights = VectorDescriptor (lenWeights);
            weightVectors = new OptimizerVectors (options.Device, vDescWeights, 0.0f);

            var vDescBiases = VectorDescriptor (outChannels);
            biasVectors = new OptimizerVectors (options.Device, vDescBiases, 0.1f);

            RandomizeWeights ((nuint)seed);

            convWtsAndBias = new MPSCnnConvolutionWeightsAndBiasesState (weightVectors.Value.Data, biasVectors.Value.Data);
            momentumVectors = NSArray<MPSVector>.FromNSObjects (weightVectors.Momentum, biasVectors.Momentum);
            velocityVectors = NSArray<MPSVector>.FromNSObjects (weightVectors.Velocity, biasVectors.Velocity);

            var odesc = new MPSNNOptimizerDescriptor (options.LearningRate, 1.0f, MPSNNRegularizationType.None, 1.0f);
            updater = new MPSNNOptimizerAdam (
                options.Device,
                beta1: 0.9f, beta2: 0.999f, epsilon: 1e-8f,
                timeStep: 0,
                optimizerDescriptor: odesc);
        }

        protected override void Dispose (bool disposing)
        {
            if (!isDisposed) {
                isDisposed = true;
            }
            base.Dispose (disposing);
        }

        [DebuggerHidden]
        public override bool Load {
            get {
                //Console.WriteLine ($"Load Conv2dDataSource {this.Label}");
                return true;
            }
        }

        public override void Purge ()
        {
            //Console.WriteLine ($"Purge Conv2dDataSource {this.Label}");
        }

        public override MPSCnnConvolutionWeightsAndBiasesState Update (IMTLCommandBuffer commandBuffer, MPSCnnConvolutionGradientState gradientState, MPSCnnConvolutionWeightsAndBiasesState sourceState)
        {
            updateCount++;

            updater.Encode (commandBuffer, gradientState, sourceState, momentumVectors, velocityVectors, convWtsAndBias);

            if (updateCount != updater.TimeStep) {
                throw new Exception ($"Update time step is out of synch");
            }

            //Console.WriteLine ($"UpdateWeights of Conv2dDataSource {this.Label}");

            return convWtsAndBias;
        }

        public Dictionary<string, float[]> GetWeights ()
        {
            return new Dictionary<string, float[]> {
                [label + ".Weights.Value"] = weightVectors.Value.ToArray (),
                //[label + ".Weights.Momentum"] = weightVectors.Momentum.ToArray(),
                //[label + ".Weights.Velocity"] = weightVectors.Velocity.ToArray(),
                [label + ".Biases.Value"] = biasVectors.Value.ToArray (),
                //[label + ".Biases.Momentum"] = biasVectors.Momentum.ToArray(),
                //[label + ".Biases.Velocity"] = biasVectors.Velocity.ToArray(),
            };
        }

        void RandomizeWeights (nuint seed)
        {
            var randomDesc = MPSMatrixRandomDistributionDescriptor.CreateUniform (-0.2f, 0.2f);
            var randomKernel = new MPSMatrixRandomMTGP32 (options.Device, MPSDataType.Float32, seed, randomDesc);

            // Run on its own buffer so as not to bother others
            using var commandBuffer = MPSCommandBuffer.Create (options.Queue);
            randomKernel.EncodeToCommandBuffer (commandBuffer, weightVectors.Value);
            commandBuffer.Commit ();
            commandBuffer.WaitUntilCompleted ();

            weightVectors.Momentum.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            weightVectors.Velocity.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            biasVectors.Value.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
            biasVectors.Momentum.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
            biasVectors.Velocity.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
        }

        public NetworkData.DataSource GetData (bool includeTrainingParameters)
        {
            var c = new NetworkData.ConvolutionDataSource {
                Weights = weightVectors.GetData (includeTrainingParameters: includeTrainingParameters),
                Biases = biasVectors.GetData (includeTrainingParameters: includeTrainingParameters),
            };
            return new NetworkData.DataSource {
                Convolution = c
            };
        }

        public void SetData (NetworkData.DataSource dataSource)
        {
            var c = dataSource.Convolution;
            if (c == null)
                return;

            weightVectors.SetData (c.Weights);
            biasVectors.SetData (c.Biases);

            weightVectors.Value.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            weightVectors.Momentum.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            weightVectors.Velocity.Data.DidModify (new NSRange (0, weightVectors.VectorByteSize));
            biasVectors.Value.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
            biasVectors.Momentum.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
            biasVectors.Velocity.Data.DidModify (new NSRange (0, biasVectors.VectorByteSize));
        }

        public bool WeightsAreValid ()
        {
            return weightVectors.WeightsAreValid () && biasVectors.WeightsAreValid ();
        }
    }

    class OptimizerVectors
    {
        public readonly int VectorLength;
        public readonly int VectorByteSize;
        public readonly MPSVectorDescriptor VectorDescriptor;
        public readonly MPSVector Value;
        public readonly MPSVector Momentum;
        public readonly MPSVector Velocity;
        public readonly IntPtr ValuePointer;

        public OptimizerVectors (IMTLDevice device, MPSVectorDescriptor descriptor, float initialValue)
        {
            VectorLength = (int)descriptor.Length;
            VectorByteSize = descriptor.ByteSize ();
            VectorDescriptor = descriptor;
            Value = Vector (device, descriptor, initialValue);
            Momentum = Vector (device, descriptor, 0.0f);
            Velocity = Vector (device, descriptor, 0.0f);
            ValuePointer = Value.Data.Contents;
        }

        public NetworkData.OptimizableVector GetData (bool includeTrainingParameters)
        {
            return new NetworkData.OptimizableVector {
                Value = GetVectorData (Value),
                Momentum = includeTrainingParameters ? GetVectorData (Momentum) : null,
                Velocity = includeTrainingParameters ? GetVectorData (Velocity) : null,
            };
        }

        public void SetData (NetworkData.OptimizableVector data)
        {
            if (data == null)
                return;
            SetVectorData (Value, data.Value);
            SetVectorData (Momentum, data.Momentum);
            SetVectorData (Velocity, data.Velocity);
        }

        NetworkData.Vector GetVectorData (MPSVector vector)
        {
            var v = new NetworkData.Vector ();
            v.Values.AddRange (vector.ToArray ());
            return v;
        }

        bool SetVectorData (MPSVector vector, NetworkData.Vector data)
        {
            if (data == null)
                return false;

            var vs = data.Values;
            var n = (int)vector.Length;
            if (n != vs.Count)
                return false;

            unsafe {
                var p = (float*)vector.Data.Contents;
                for (var i = 0; i < n; i++) {
                    *p++ = vs[i];
                }
            }
            return true;
        }

        public void Synchronize (IMTLCommandBuffer commandBuffer)
        {
            Value.Synchronize (commandBuffer);
            Momentum.Synchronize (commandBuffer);
            Velocity.Synchronize (commandBuffer);
        }

        public bool WeightsAreValid ()
        {
            return Value.IsValid () && Momentum.IsValid () && Velocity.IsValid ();
        }
    }
}
