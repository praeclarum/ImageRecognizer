#nullable enable

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading.Tasks;
using Foundation;
using Metal;
using MetalPerformanceShaders;

namespace ImageRecognizerLibrary
{
    public class RecognizerNetwork : Network
    {
        const int BatchSize = 40;
        const int NumTrainingIterations = 300;

        readonly MPSNNFilterNode inferenceNodesTail;
        readonly MPSNNGraph inferenceGraph;
        readonly MPSNNFilterNode trainingNodesTail;
        readonly MPSNNFilterNode[] lossExitPoints;
        readonly MPSNNGraph trainingGraph;

        static readonly MnistDataSet dataSet = new MnistDataSet (42);

        public RecognizerNetwork ()
        {
            trainingNodesTail = CreateTrainingNodes ();
            lossExitPoints = trainingNodesTail.GetTrainingGraph (gradientImageSource: null, nodeHandler: (gr, inf, infs, grs) => {
                gr.ResultImage.Format = MPSImageFeatureChannelFormat.Float32;
            });
            Console.WriteLine (string.Join ("\n", lossExitPoints.Select ((x, i) => $"T{i}: {x.GetType ()}")));
            trainingGraph = new MPSNNGraph (device, lossExitPoints[0].ResultImage, resultIsNeeded: true) {
                Format = FeatureChannelFormat,
            };

            inferenceNodesTail = CreateInferenceNodes ();
            inferenceGraph = new MPSNNGraph (device, inferenceNodesTail.ResultImage, resultIsNeeded: true) {
                Format = FeatureChannelFormat,
            };
        }

        public override MPSNNGraph[] Graphs {
            get => new[] {
                inferenceGraph,
                trainingGraph
            };
        }

        MPSNNFilterNode CreateInferenceNodes () => CreateNodes (false);
        MPSNNFilterNode CreateTrainingNodes () => CreateNodes (true);

        MPSNNFilterNode CreateNodes (bool isTraining)
        {
            var input = new MPSNNImageNode (null);

            var conv1 = new MPSCnnConvolutionNode (input, Conv2d (1, 32, kernelSize: 5, label: "Conv1")) {
                PaddingPolicy = sameConvPadding
            };
            var relu1 = new MPSCnnNeuronReLUNode (conv1.ResultImage, a: 0.0f);
            var pool1 = new MPSCnnPoolingMaxNode (relu1.ResultImage, size: 2, stride: 2) {
                PaddingPolicy = samePoolingPadding
            };

            var conv2 = new MPSCnnConvolutionNode (pool1.ResultImage, Conv2d (32, 64, kernelSize: 5, label: "Conv2")) {
                PaddingPolicy = sameConvPadding
            };
            var relu2 = new MPSCnnNeuronReLUNode (conv2.ResultImage, a: 0.0f);
            var pool2 = new MPSCnnPoolingMaxNode (relu2.ResultImage, size: 2, stride: 2) {
                PaddingPolicy = samePoolingPadding
            };

            var fc1 = new MPSCnnFullyConnectedNode (pool2.ResultImage, Conv2d (64, 1024, kernelSize: 7, label: "Dense1"));
            var relu3 = new MPSCnnNeuronReLUNode (fc1.ResultImage, a: 0.0f);

            MPSNNFilterNode fc2Input = relu3;
            if (isTraining) {
                var dropNode = new MPSCnnDropoutNode (relu3.ResultImage, keepProbability: 0.5f, seed: 42, maskStrideInPixels: new MTLSize (1, 1, 1));
                fc2Input = dropNode;
            }

            var fc2 = new MPSCnnFullyConnectedNode (fc2Input.ResultImage, Conv2d (1024, 10, kernelSize: 1, label: "Dense2"));

            if (isTraining) {
                var lossDesc = MPSCnnLossDescriptor.Create (MPSCnnLossType.SoftMaxCrossEntropy, MPSCnnReductionType.Sum);
                lossDesc.Weight = 1.0f / BatchSize;

                var loss = new MPSCnnLossNode (fc2.ResultImage, lossDesc);
                return loss;
            }
            else {
                var sft = new MPSCnnSoftMaxNode (fc2.ResultImage);
                return sft;
            }
        }

        protected override async Task PredictBatchesAsync ()
        {
            for (; ; ) {
                await PredictBatchAsync ().ConfigureAwait (false);
                await Task.Delay (1000);
            }
        }

        Task PredictBatchAsync ()
        {
            return Task.Run (PredictBatch);
        }

        void PredictBatch ()
        {
            var (inputs, losses) = dataSet.GetRandomBatch (device, BatchSize);

            var inputDesc = MPSImageDescriptor.GetImageDescriptor (
                MPSImageFeatureChannelFormat.Unorm8,
                MnistDataSet.ImageSize, MnistDataSet.ImageSize, 1, 1, MTLTextureUsage.ShaderRead);

            var inputBatch = new List<MPSImage> ();

            using var commandBuffer = MPSCommandBuffer.Create (compileOptions.Queue);

            var outputBatch = inferenceGraph.EncodeBatch (commandBuffer, new[] { inputs }, null);
            MPSImageBatch.Synchronize (outputBatch, commandBuffer);

            commandBuffer.Commit ();
            commandBuffer.WaitUntilCompleted ();

            ShowImage (inputs[0]);
            ShowOutputImage (outputBatch[0]);
        }

        protected override async Task TrainBatchesAsync ()
        {
            DumpWeights ();

            for (var i = 0; i < NumTrainingIterations; i++) {
                Console.WriteLine ($"Training Batch {i}/{NumTrainingIterations} ({BatchSize} images each)");
                await Task.Run (TrainBatch).ConfigureAwait (false);

                //Console.WriteLine ($"Done training: {outputImages.Count} Outputs");
                inferenceGraph.ReloadFromDataSources ();
                await PredictBatchAsync ();
            }

            DumpWeights ();
        }

        void TrainBatch ()
        {
            var (inputs, losses) = dataSet.GetRandomBatch (device, BatchSize);

            using var commandBuffer = MPSCommandBuffer.Create (compileOptions.Queue);

            var returnBatch = trainingGraph.EncodeBatch (commandBuffer, new[] { inputs }, new[] { losses });
            MPSImageBatch.Synchronize (returnBatch, commandBuffer);

            var lossOuts = new List<MPSImage> ();
            for (var i = 0; i < BatchSize; i++) {
                lossOuts.Add (((MPSCnnLossLabels)losses[i]).LossImage);
            }
            MPSImageBatch.Synchronize (NSArray<MPSImage>.FromNSObjects (lossOuts.ToArray ()), commandBuffer);

            commandBuffer.Commit ();
            commandBuffer.WaitUntilCompleted ();

            var loss = ReduceLoss (lossOuts);
            Console.WriteLine ($"LOSS = {loss}");
            //ShowImage (inputs[0]);
        }

        unsafe float ReduceLoss (IList<MPSImage> batch)
        {
            float ret = 0;
            var val = stackalloc float[1];

            for (int i = 0; i < batch.Count; i++) {
                var curr = batch[i];
                *val = 0;
                //assert (curr.width * curr.height * curr.featureChannels == 1);
                curr.ReadBytes ((IntPtr)val, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);
                ret += val[0] / (float)BatchSize;
            }
            return ret;
        }
    }
}
