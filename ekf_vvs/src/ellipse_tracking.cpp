//
// Created by valentinohx on 17/05/19.
// subscribe to "/fkuka_lwr4p/camera/image_raw"
// publish to "/ellipse_tracking/ellipses_position"

#include "ellipse_detection/ellipse_detection.h"
#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include <cv_bridge/cv_bridge.h>

#include <visp3/blob/vpDot2.h>
#include <visp3/core/vpImage.h>
#include <visp3/core/vpImagePoint.h>
#include <visp3/core/vpIoTools.h>
#include <visp3/gui/vpDisplayGDI.h>
#include <visp3/gui/vpDisplayGTK.h>
#include <visp3/gui/vpDisplayOpenCV.h>
#include <visp3/gui/vpDisplayX.h>
#include <visp3/io/vpImageIo.h>
#include <visp3/io/vpParseArgv.h>
#include <visp3/core/vpConfig.h>
#include <visp3/core/vpDebug.h>
#include "std_msgs/Float64MultiArray.h"
#include "camera_pixel_msg/StampedPixelArray.h"


bool isFirstFrame = true;
bool opt_display  = false;
bool dispalyInit = true;

cv::Mat cvImg;
int n_ellipse = 4;
vpImage<unsigned char> srcImg;

vpDot2 d[4];
vpImagePoint cog;  //center of gravity
vpImagePoint ci;
std::vector<double> msg_vec;
//std_msgs::Float64MultiArray ellipseUV_msg;
camera_pixel_msg::StampedPixelArray ellipseUV_msg;
vpDisplayX display;
ros::Publisher *uv_publisher;

void imageCallback(const sensor_msgs::ImageConstPtr& msg)
{
    try {
        cvImg = cv_bridge::toCvShare(msg, "bgr8")->image;
        vpImageConvert::convert(cvImg, srcImg);
        if (isFirstFrame)
        {
            if (opt_display)
            {
                if (dispalyInit)
                {
                    display.init(srcImg, 100, 100, "Display...");
                    dispalyInit = false;
                }
                vpDisplay::display(srcImg);
                vpDisplay::flush(srcImg);
                for (int di = 0; di < n_ellipse; di++) {
                    d[di].setGraphics(true);
                }
            }
            else
            {
                for (int di = 0; di<n_ellipse; di++)
                {
                    d[di].setGraphics(false);
                }
            }
            //detect ellipse in raw image
            std::vector<Ellipse> ellipse = ellipsoid_OnImage(cvImg);

            //filter out the ellipses that is too close
            std::vector<Ellipse> ellipse_refined;
            double ellipse_dist_threshold = 40;
            while(!ellipse.empty())
            {
                Ellipse first_ellipse = ellipse.front();
                auto it = ellipse.begin();
                while ( it != ellipse.end() )
                {
                    if ( cv::norm( cv::Point2f(first_ellipse._xc, first_ellipse._yc) - cv::Point2f((*it)._xc, (*it)._yc)) < ellipse_dist_threshold )
                    {
                        first_ellipse._xc = ( first_ellipse._xc + (*it)._xc ) / 2;
                        first_ellipse._yc = ( first_ellipse._yc + (*it)._yc ) / 2;
                        ellipse.erase( it );
                    }
                    else
                        it++;
                }
                ellipse_refined.push_back(first_ellipse);
            }

            if (ellipse_refined.size() == n_ellipse)
            {
                //sort the detected ellipse in clockwise way
                std::vector<cv::Point2f> feature;
                feature.clear();
                for(int i = 0; i < n_ellipse; i++)
                {
                    feature.emplace_back(ellipse_refined[i]._xc, ellipse_refined[i]._yc);
                }

                cv::Point2f accP(0.0, 0.0);
                for (const auto & fi : feature)
                {
                    accP += fi;
                }
                cv::Point2f centerP(accP.x /( feature.size()), accP.y /( feature.size()) );
                std::multimap< float, cv::Point2f> Point_Atan2;
                for (auto & fi : feature)
                {
                    float PAtan2 = atan2(fi.y - centerP.y, fi.x - centerP.x);
                    Point_Atan2.insert ( std::pair<float,cv::Point2f>(PAtan2 ,fi) );
                }
                std::vector<cv::Point2f> featureSorted;
                for(auto & it : Point_Atan2)
                {
                    featureSorted.push_back(it.second);
                }
                msg_vec.clear();
                for (int i = 0; i < n_ellipse; i++)
                {
                    d[i].setComputeMoments(true);
                    d[i].setGrayLevelPrecision(0.85);

                    ci.set_u(featureSorted[i].x);  //take care the u,v order
                    ci.set_v(featureSorted[i].y);

                    //initialize the tracker
                    d[i].initTracking(srcImg, ci);

                    msg_vec.push_back(featureSorted[i].x);
                    msg_vec.push_back(featureSorted[i].y);
                }
                isFirstFrame = false;

                ellipseUV_msg.header.frame_id = 7;
                ellipseUV_msg.header.seq = 7;
                ellipseUV_msg.header.stamp = ros::Time::now();

                ellipseUV_msg.data.clear();
                ellipseUV_msg.data.insert(ellipseUV_msg.data.end(), msg_vec.begin(), msg_vec.end());
                uv_publisher->publish(ellipseUV_msg);
            }
        }
        else
        {
            if(opt_display) vpDisplay::display(srcImg);
            msg_vec.clear();
            for (int di = 0; di<n_ellipse; di++)
            {
                d[di].track(srcImg);
                cog = d[di].getCog();

                msg_vec.push_back(cog.get_u());
                msg_vec.push_back(cog.get_v());
                if(opt_display) {
                    vpDisplay::displayCross(srcImg, cog, 10, vpColor::green);
                    vpDisplay::displayCharString(srcImg, cog, std::to_string(di+1).c_str(), vpColor::red);
                }
            }

            ellipseUV_msg.header.frame_id = 7;
            ellipseUV_msg.header.seq = 1;
            ellipseUV_msg.header.stamp = ros::Time::now();

            ellipseUV_msg.data.clear();
            ellipseUV_msg.data.insert(ellipseUV_msg.data.end(), msg_vec.begin(), msg_vec.end());
            uv_publisher->publish(ellipseUV_msg);

            if(opt_display) {
                vpDisplay::displayCross(srcImg, vpImagePoint(400, 400), 400, vpColor::red);
//                vpDisplay::displayCross(srcImg, vpImagePoint(324.23, 324.23), 20, vpColor::red);
//                vpDisplay::displayCross(srcImg, vpImagePoint(476.78, 324.23), 20, vpColor::red);
//                vpDisplay::displayCross(srcImg, vpImagePoint(476.78, 476.78), 20, vpColor::red);
//                vpDisplay::displayCross(srcImg, vpImagePoint(324.23, 476.78), 20, vpColor::red);

                vpDisplay::displayCross(srcImg, vpImagePoint(362.36376, 362.36376), 20, vpColor::red);
                vpDisplay::displayCross(srcImg, vpImagePoint(438.63624, 362.36376), 20, vpColor::red);
                vpDisplay::displayCross(srcImg, vpImagePoint(438.63624, 438.63624), 20, vpColor::red);
                vpDisplay::displayCross(srcImg, vpImagePoint(362.36376, 438.63624), 20, vpColor::red);

                vpDisplay::flush(srcImg);
            }
        }
    }
    catch (cv_bridge::Exception& e)
    {
        ROS_ERROR("Could not convert from '%s' to 'bgr8'.", msg->encoding.c_str());
    }
}

int main(int argc, char** argv)
{
    ros::init(argc, argv, "ellipse_tracker");
    ros::NodeHandle nh;
    image_transport::ImageTransport it(nh);
    image_transport::Subscriber sub = it.subscribe("/fkuka_lwr4p/camera/image_raw", 1, imageCallback);
//    ros::Publisher uv_pub = nh.advertise<std_msgs::Float64MultiArray>("/ellipse_tracking/ellipses_position", 64);
    ros::Publisher uv_pub = nh.advertise<camera_pixel_msg::StampedPixelArray>("/ellipse_tracking/ellipses_position", 64);
    uv_publisher = &uv_pub;
    ros::spin();
    return 0;
}

