#nullable enable

using System;
using System.Runtime.InteropServices;
using Foundation;
using Metal;
using MetalPerformanceShaders;

namespace ImageRecognizerLibrary
{
    public static class MPSExtensions
    {
        public static MPSVectorDescriptor VectorDescriptor (int length, MPSDataType dataType = MPSDataType.Float32) =>
            MPSVectorDescriptor.Create ((nuint)length, dataType);

        public static MPSVector Vector (IMTLDevice device, MPSVectorDescriptor descriptor, float initialValue)
        {
            var v = new MPSVector (device, descriptor);
            var vectorByteSize = GetByteSize (descriptor);
            unsafe {
                float biasInit = initialValue;
                var biasInitPtr = (IntPtr)(float*)&biasInit;
                memset_pattern4 (v.Data.Contents, biasInitPtr, vectorByteSize);
            }
            return v;
        }
        [System.Runtime.InteropServices.DllImport (@"__Internal", CallingConvention = System.Runtime.InteropServices.CallingConvention.Cdecl)]
        static extern void memset_pattern4 (IntPtr b, IntPtr pattern4, nint len);

        public static float[] ToArray (this MPSVector vector)
        {
            var ar = new float[vector.Length];
            Marshal.Copy (vector.Data.Contents, ar, 0, ar.Length);
            return ar;
        }

        public static bool IsValid (this MPSVector vector)
        {
            var ar = vector.ToArray ();
            for (var i = 0; i < ar.Length; i++) {
                var v = ar[i];
                if (float.IsNaN (v))
                    return false;
                if (float.IsInfinity (v))
                    return false;
                if (float.IsNegativeInfinity (v))
                    return false;
            }
            return true;
        }

        public static int GetByteSize (this MPSVectorDescriptor descriptor) =>
            (int)descriptor.Length * GetByteSize (descriptor.DataType);

        public static int GetByteSize (this MPSDataType dataType) =>
            dataType switch
            {
                MPSDataType.Unorm8 => 1,
                MPSDataType.Float32 => 4,
                var x => throw new NotSupportedException ($"Cannot get size of {x}")
            };
#if __IOS__
        public static void DidModify (this IMTLBuffer buffer, NSRange range)
        {
        }
#endif
    }
}
