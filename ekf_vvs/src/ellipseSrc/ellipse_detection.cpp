
#include<ellipse_detection/ellipse_detection.h>

 vector<Ellipse> ellipsoid_OnImage(cv::Mat &image)
{
    cv::Size sz = image.size();
    // Convert to grayscale
    cv::Mat1b gray;
    cv::cvtColor(image, gray, CV_BGR2GRAY);

    // Parameters Settings (Sect. 4.2)
    int		iThLength = 16;
    float	fThObb = 3.0f;
    float	fThPos = 1.0f;
    float	fTaoCenters = 0.05f;
    int 	iNs = 16; //16 by default
    float	fMaxCenterDistance = sqrt(float(sz.width*sz.width + sz.height*sz.height)) * fTaoCenters;
    float	fThScoreScore = 0.4f;

    // Other constant parameters settings.
    // Gaussian filter parameters, in pre-processing
    cv::Size	szPreProcessingGaussKernelSize = Size(5, 5);
    double	dPreProcessingGaussSigma = 1.0;

    float	fDistanceToEllipseContour = 0.1f;	// (Sect. 3.3.1 - Validation)
    float	fMinReliability = 0.4f;	// Const parameters to discard bad ellipses

    // Initialize Detector with selected parameters
    CEllipseDetectorYaed* yaed = new CEllipseDetectorYaed();
    yaed->SetParameters(szPreProcessingGaussKernelSize,
                        dPreProcessingGaussSigma,
                        fThPos,
                        fMaxCenterDistance,
                        iThLength,
                        fThObb,
                        fDistanceToEllipseContour,
                        fThScoreScore,
                        fMinReliability,
                        iNs
                       );
    // Detect
    vector<Ellipse> ellsYaed;
    yaed->Detect(gray, ellsYaed);
    return ellsYaed;
}
