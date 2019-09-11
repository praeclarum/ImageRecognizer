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
using static ImageRecognizerLibrary.MPSExtensions;

namespace ImageRecognizerLibrary
{
    public static class ImageConversion
    {
        public static unsafe UIKit.UIImage GetUIImage (MPSImage mpsImage)
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
                        *p++ = ClampRGBA32Float (data[y * (width * 3) + x * 3 + 2]);
                        *p++ = ClampRGBA32Float (data[y * (width * 3) + x * 3 + 1]);
                        *p++ = ClampRGBA32Float (data[y * (width * 3) + x * 3 + 0]);
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
    }
}
