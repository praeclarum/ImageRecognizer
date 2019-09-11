# ImageRecognizer

This is a demo app that uses Metal Performance Shaders, and particularly, MPSNNGraph,
to train a neural network to recognize digits.

It's written in C# using Xamarin and is roughly a port of Apple's MPSTrainingClassifier sample.

**Requires iOS 13**

## Important Classes

* `RecognizerNetwork` constructs the `MPSNNGraph` needed to train and execute
neural networks. It constructs a deep convolutional network and expects
to work on black and white images sized 28x28.

* `MnistDataSet` loads the mnist data set for training and for testing.

* `ConvolutionWeights` implements data storage for MPS nodes.
Objects of this type hold the weights that the network has learned.

* `NetworkData` namespace contains protocol buffers to save the
weights used by `ConvolutionWeights`.

* `ImageConversion` renders `MPSImages` to `UIImages` for display.

* `MPSExtensions` contains helper functions simplify coding in MPS.


