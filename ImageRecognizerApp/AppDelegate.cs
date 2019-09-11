using Foundation;
using UIKit;

namespace ImageRecognizerApp
{
    [Register ("AppDelegate")]
    public class AppDelegate : UIApplicationDelegate
    {
        public override UIWindow Window {
            get;
            set;
        }

        public override bool FinishedLaunching (UIApplication application, NSDictionary launchOptions)
        {
            var vc = new MainViewController ();
            var nav = new UINavigationController (vc);

            Window = new UIWindow (UIScreen.MainScreen.Bounds);
            Window.RootViewController = nav;
            Window.MakeKeyAndVisible ();

            return true;
        }
    }
}

