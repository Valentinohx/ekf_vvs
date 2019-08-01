
#include "opencv2/core.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d.hpp>
#include <opencv2/imgcodecs.hpp>
#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/videoio.hpp"


#include <fstream>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iostream>

#include "ellipse_detection/EllipseDetectorYaed.h"

vector<Ellipse> ellipsoid_OnImage(cv::Mat &image);
