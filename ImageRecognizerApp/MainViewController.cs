using System;
using System.IO;
using System.Threading.Tasks;
using UIKit;
using EasyLayout;
using ImageRecognizerLibrary;

namespace ImageRecognizerApp
{
    public class MainViewController : UIViewController
    {
        readonly UITextView textView = new UITextView ();
        readonly UIImageView imageView = new UIImageView {
            ContentMode = UIViewContentMode.ScaleAspectFit,
            BackgroundColor = UIColor.Clear,
        };
        readonly UIImageView imageOutputView = new UIImageView {
            ContentMode = UIViewContentMode.ScaleAspectFit,
            BackgroundColor = UIColor.Clear,
        };

        public MainViewController ()
        {
            Title = "Image Recognizer";

            textView.Font = UIFont.FromName ("Menlo-Regular", 14);
            textView.AlwaysBounceVertical = true;
            textView.Editable = false;

            //
            // Route all console text to the textView
            //
            var w = new TextViewWriter (textView, Console.Out);
            Console.SetOut (w);
            Console.SetError (w);
            Console.WriteLine ("Hello");
        }

        public override async void ViewDidLoad ()
        {
            base.ViewDidLoad ();

            textView.Frame = View.Bounds;
            textView.AutoresizingMask = UIViewAutoresizing.FlexibleDimensions;

            View.AddSubview (textView);
            View.AddSubview (imageView);
            View.AddSubview (imageOutputView);

            View.ConstrainLayout (() =>
                imageView.Frame.Right == View.Frame.Right &&
                imageView.Frame.Top == View.Frame.Top + 100 &&
                imageView.Frame.Width == 256 &&
                imageView.Frame.Height == 256 &&
                imageOutputView.Frame.Right == imageView.Frame.Right &&
                imageOutputView.Frame.Top == imageView.Frame.Bottom &&
                imageOutputView.Frame.Width == imageView.Frame.Width &&
                imageOutputView.Frame.Height == imageView.Frame.Height
                );

            View.BackgroundColor = UIColor.SystemBackgroundColor;

            await TrainNetworkAsync ();
        }

        async Task TrainNetworkAsync ()
        {
            try {
                var weightsName = "mnist5.weights";
                var weightsPath = Path.Combine (Environment.GetFolderPath (Environment.SpecialFolder.MyDocuments), weightsName);
                var hasWeights = File.Exists (weightsPath);
                bool needsTrain = !hasWeights;
                needsTrain = true;

                //
                // Create the network
                //
                var network = new RecognizerNetwork ();
                network.ImagesShown += Network_ImagesShown;
                Console.WriteLine (network);

                //
                // Read previously trained weights
                //
                if (hasWeights) {
                    await network.ReadAsync (weightsPath);
                }

                //
                // Load the data set
                //
                var dataSet = new MnistDataSet (seed: 42);

                //
                // Train the network
                //
                if (needsTrain)
                    await network.TrainAsync (dataSet);

                //
                // Save the network if training went well
                //
                if (network.WeightsAreValid ()) {
                    if (needsTrain)
                        await network.WriteAsync (weightsPath);

                    //
                    // Start predicting
                    //
                    await network.PredictAsync (dataSet);
                }
                else {
                    Console.WriteLine ("Bad weights");
                }

                //
                // All done
                //
                network.ImagesShown -= Network_ImagesShown;
            }
            catch (Exception ex) {
                Console.WriteLine (ex);
            }
        }

        void Network_ImagesShown ((UIImage InputImage, UIImage OutputImage) images)
        {
            BeginInvokeOnMainThread (() => {
                imageView.Image = images.InputImage;
                imageOutputView.Image = images.OutputImage;
            });
        }
    }
}
