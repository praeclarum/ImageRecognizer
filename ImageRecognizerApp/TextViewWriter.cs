#nullable enable

using System.IO;
using System.Text;
using UIKit;

namespace ImageRecognizerApp
{
    /// <summary>
    /// A System.IO.TextWriter that outputs to a UITextView
    /// and forwards the writes to another TextWriter
    /// </summary>
    class TextViewWriter : TextWriter
    {
        readonly TextWriter previousWriter;
        readonly UITextView textView;

        public TextViewWriter(UITextView textView, TextWriter previousWriter)
        {
            this.textView = textView;
            this.previousWriter = previousWriter;
        }

        public override Encoding Encoding => Encoding.UTF8;

        public override void Write (char value)
        {
            previousWriter.Write (value);
            textView.BeginInvokeOnMainThread (() => {
                var newText = textView.Text + value;
                textView.Text = newText;
                textView.ScrollRangeToVisible (new Foundation.NSRange (newText.Length, 0));
            });
        }

        public override void Write (string value)
        {
            previousWriter.Write (value);
            textView.BeginInvokeOnMainThread (() => {
                var newText = textView.Text + value;
                textView.Text = newText;
                textView.ScrollRangeToVisible (new Foundation.NSRange (newText.Length, 0));
            });
        }

        public override void WriteLine (string value)
        {
            previousWriter.WriteLine (value);
            textView.BeginInvokeOnMainThread (() => {
                var newText = textView.Text + value + "\n";
                textView.Text = newText;
                textView.ScrollRangeToVisible (new Foundation.NSRange (newText.Length, 0));
            });
        }
    }
}
