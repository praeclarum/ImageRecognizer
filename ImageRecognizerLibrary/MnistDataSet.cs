using System;
using System.IO;
using System.IO.Compression;
using Foundation;
using MetalPerformanceShaders;
using Metal;
using System.Collections.Generic;

namespace ImageRecognizerLibrary
{
    public class MnistDataSet
    {
        readonly int numImages;
        readonly byte[] imagesData;
        readonly byte[] labelsData;
        readonly Random random;

        public const int ImageSize = 28;
        const int ImagesPrefixSize = 16;
        const int LabelsPrefixSize = 8;

        public MnistDataSet (int seed)
        {
            random = new Random (seed);
            imagesData = ReadGZip (NSBundle.MainBundle.PathForResource ("mnist-images", "gz"));
            labelsData = ReadGZip (NSBundle.MainBundle.PathForResource ("mnist-labels", "gz"));
            numImages = labelsData.Length - LabelsPrefixSize;
        }

        static byte[] ReadGZip (string path)
        {
            using var fs = File.OpenRead (path);
            using var gz = new GZipStream (fs, CompressionMode.Decompress);
            using var memoryStream = new MemoryStream ();
            gz.CopyTo (memoryStream);
            return memoryStream.ToArray ();
        }

        public (NSArray<MPSImage> Sources, NSArray<MPSState> Targets) GetRandomBatch (IMTLDevice device, int batchSize)
        {
            var trainImageDesc = MPSImageDescriptor.GetImageDescriptor (
                MPSImageFeatureChannelFormat.Unorm8,
                ImageSize, ImageSize, 1,
                1,
                MTLTextureUsage.ShaderWrite | MTLTextureUsage.ShaderRead);

            var trainBatch = new List<MPSImage> ();
            var lossStateBatch = new List<MPSState> ();

            unsafe {
                fixed (byte* imagesPointer = imagesData)
                fixed (byte* labelsPointer = labelsData) {

                    for (var i = 0; i < batchSize; i++) {
                        var randomIndex = random.Next (numImages);

                        var trainImage = new MPSImage (device, trainImageDesc) {
                            Label = "TrainImage" + i
                        };
                        trainBatch.Add (trainImage);
                        var trainImagePointer = imagesPointer + ImagesPrefixSize + randomIndex * ImageSize * ImageSize;
                        trainImage.WriteBytes ((IntPtr)trainImagePointer, MPSDataLayout.HeightPerWidthPerFeatureChannels, 0);

                        var labelPointer = labelsPointer + LabelsPrefixSize + randomIndex;
                        var labelsValues = new float[12];
                        labelsValues[*labelPointer] = 1;

                        fixed (void* p = labelsValues) {
                            using var data = NSData.FromBytes ((IntPtr)p, 12 * sizeof(float));
                            var desc = MPSCnnLossDataDescriptor.Create (
                                data, MPSDataLayout.HeightPerWidthPerFeatureChannels, new MTLSize (1, 1, 12));
                            var lossState = new MPSCnnLossLabels (device, desc);
                            lossStateBatch.Add (lossState);
                        }
                    }
                }
            }

            return (NSArray<MPSImage>.FromNSObjects (trainBatch.ToArray ()),
                    NSArray<MPSState>.FromNSObjects (lossStateBatch.ToArray ()));
        }
    }
}
