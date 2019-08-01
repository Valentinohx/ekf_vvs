
#include "ekf_vvs/ekf_base.hpp"
#include <ekf_vvs/VS_with_sensor_fusion.hpp>

#include<ros/ros.h>
#include <ctime>
#include <ratio>
#include <chrono>
#include <iostream>

#include <Eigen/Dense>
#include <Eigen/SVD>

#include <std_msgs/Float64MultiArray.h>
#include <std_msgs/Float32MultiArray.h>
#include <sensor_msgs/JointState.h>
#include <control_blocks_msgs/ArrayStamped.h>
#include <camera_pixel_msg/StampedPixelArray.h>

#include <tf/transform_listener.h>
#include <tf/tfMessage.h>

#include <visp3/gui/vpPlot.h>
#include <visp3/robot/vpSimulatorCamera.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/vs/vpServo.h>

#include <visp3/core/vpConfig.h>

//perform pseudo inverse
template<typename _Matrix_Type_>
_Matrix_Type_ pseudoInverse(const _Matrix_Type_ &a, double epsilon = std::numeric_limits<double>::epsilon())
{
    Eigen::JacobiSVD< _Matrix_Type_ > svd(a ,Eigen::ComputeThinU | Eigen::ComputeThinV);
    double tolerance = epsilon * std::max(a.cols(), a.rows()) *svd.singularValues().array().abs()(0);
    return svd.matrixV() *  (svd.singularValues().array().abs() > tolerance).select(svd.singularValues().array().inverse(), 0).matrix().asDiagonal() * svd.matrixU().adjoint();
}

typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXs;
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

VectorXs featurexy(8), errorZ(4), featureError(8), featureDepth(4), jointStates(7);

//camera parameters
double px = 476.703;
double py = 476.703;
double u0 = 400.5;
double v0 = 400.5;

int cameraFlag = 0;
int encoderFlag = 0;

bool cameraAvaliable = true;
bool encoderAvaliable = true;

ros::Time lastCameraAvaliableTime, lastEncoderAvaliableTime;

void jointCallback(const sensor_msgs::JointState& msg)
{
    encoderFlag++;
    for( int i=0; i< 7; i++)
    {
        jointStates(i) = msg.position[i];
    }
}

void featureCallback(const camera_pixel_msg::StampedPixelArray& msg)
{
    cameraFlag++;
    for( int i=0; i < 4; i++)
    {
        featurexy(2*i)     = (msg.data[2*i] - u0)/px;
        featurexy(2*i + 1) = (msg.data[2*i + 1] - v0)/py;
    }
}



int main(int argc, char** argv) {
    ros::init(argc, argv, "VS_joints_sim");
    ros::NodeHandle nh;

    //subscriber and publisher
    ros::Subscriber joint_states_sub = nh.subscribe("/joint_states", 1, jointCallback);
    ros::Subscriber feature_sub = nh.subscribe("/ellipse_tracking/ellipses_position", 1, featureCallback);
//    ros::Publisher vel_cmd_pub = nh.advertise<std_msgs::Float32MultiArray>("/KUKA_LWR4P/vel_cmd", 1000);    //sub pub
    ros::Publisher vel_cmd_pub = nh.advertise<control_blocks_msgs::ArrayStamped>("/fkuka_lwr4p/velocity_controller/velocity", 1000);    //sub pub
    std::string urdf;
    if(!nh.getParam("robot_description", urdf)) {
        ROS_ERROR("Failed to retrieve parameter 'robot_description'");
        return 0;
    }

    std::string endEffectorName;
    if(!nh.getParam("endEffectorName", endEffectorName)) {
        ROS_ERROR("Failed to retrieve parameter 'endEffectorName'");
        return 0;
    }

    std::string modelPath;
    if(!nh.getParam("modelPath", modelPath))
    {
        ROS_ERROR("Failed to retrieve model Path parameter 'modelPath'");
        return 0;
    }

    //get camera frame from parameter server
    std::string camFrame;
    if(!nh.getParam("camFrame", camFrame)) {
        ROS_ERROR("Failed to retrieve parameter 'camFrame'");
        return 0;
    }

    // retrieve visual servoing parameters
    double  lamda;
    if(!nh.getParam("lamda", lamda)) {
        ROS_ERROR("Failed to retrieve parameter 'lamda'");
        return 0;
    }

    double Z_star;
    if(!nh.getParam("Z_star", Z_star)) {
        ROS_ERROR("Failed to retrieve parameter 'Z_star'");
        return 0;
    }

    double dt;
    if(!nh.getParam("dt", dt)) {
        ROS_ERROR("Failed to retrieve parameter 'dt'");
        return 0;
    }

    //ekf parameters
    double  alpha_xy, alpha_q, alpha_Z,  beta_q,
            gama_xy, gama_q, pp_xy, pp_q, pp_Z;

    if(!nh.getParam("alpha_xy", alpha_xy)) {
        ROS_ERROR("Failed to retrieve parameter 'alpha_xy'");
        return 0;
    }
    if(!nh.getParam("alpha_q", alpha_q)) {
        ROS_ERROR("Failed to retrieve parameter 'alpha_q'");
        return 0;
    }
    if(!nh.getParam("alpha_Z", alpha_Z)) {
        ROS_ERROR("Failed to retrieve parameter 'alpha_Z'");
        return 0;
    }
    if(!nh.getParam("beta_q", beta_q)) {
        ROS_ERROR("Failed to retrieve parameter 'beta_q'");
        return 0;
    }
    if(!nh.getParam("gama_xy", gama_xy)) {
        ROS_ERROR("Failed to retrieve parameter 'gama_xy'");
        return 0;
    }
    if(!nh.getParam("gama_q", gama_q)) {
        ROS_ERROR("Failed to retrieve parameter 'gama_q'");
        return 0;
    }
    if(!nh.getParam("pp_xy", pp_xy)) {
        ROS_ERROR("Failed to retrieve parameter 'pp_xy'");
        return 0;
    }
    if(!nh.getParam("pp_q", pp_q)) {
        ROS_ERROR("Failed to retrieve parameter 'pp_q'");
        return 0;
    }
    if(!nh.getParam("pp_Z", pp_Z)) {
        ROS_ERROR("Failed to retrieve parameter 'pp_Z'");
        return 0;
    }

    int ZInLoop;
    if(!nh.getParam("ZInLoop", ZInLoop)) {
        ROS_ERROR("Failed to retrieve parameter 'ZInLoop'");
        return 0;
    }

    int xyInLoop;
    if(!nh.getParam("xyInLoop", xyInLoop)) {
        ROS_ERROR("Failed to retrieve parameter 'xyInLoop'");
        return 0;
    }

    int itermax;
    if(!nh.getParam("itermax", itermax)) {
        ROS_ERROR("Failed to retrieve parameter 'itermax'");
        return 0;
    }    //get  parameters

    pinocchio::Model model;
    pinocchio::urdf::buildModel(modelPath, model, false);
    pinocchio::Data  data(model);
    const unsigned int nP = 4;   // 4 points

    VS_with_sensor_fusion<double> ekf_task(nP, dt, model);    //construct ekf

    tf::TransformListener listener;
    tf::StampedTransform Tmm;
    tf::StampedTransform Tpm;
    tf::StampedTransform Tmp;
    tf::StampedTransform Tpp;

    //find out the transformation matrix tf between boardbw and camera_link C^T_O
    listener.waitForTransform(camFrame,  "/boardbw_fix_circ_mm",  ros::Time(0), ros::Duration(3.0));
    listener.lookupTransform(camFrame,   "/boardbw_fix_circ_mm",  ros::Time(0), Tmm);

    listener.waitForTransform(camFrame,  "/boardbw_fix_circ_pm",  ros::Time(0), ros::Duration(3.0));
    listener.lookupTransform(camFrame,   "/boardbw_fix_circ_pm",  ros::Time(0), Tpm);

    listener.waitForTransform(camFrame,  "/boardbw_fix_circ_mp",  ros::Time(0), ros::Duration(3.0));
    listener.lookupTransform(camFrame,   "/boardbw_fix_circ_mp",  ros::Time(0), Tmp);

    listener.waitForTransform(camFrame,  "/boardbw_fix_circ_pp",  ros::Time(0), ros::Duration(3.0));
    listener.lookupTransform( camFrame,  "/boardbw_fix_circ_pp",  ros::Time(0), Tpp);

    //label order: pm->1 ; mm->2; mp->3, pp->4
    //in camera frame
    featurexy<<Tpm.getOrigin().x()/Tpm.getOrigin().z(),  Tpm.getOrigin().y()/Tpm.getOrigin().z(),
            Tmm.getOrigin().x()/Tmm.getOrigin().z(),  Tmm.getOrigin().y()/Tmm.getOrigin().z(),
            Tmp.getOrigin().x()/Tmp.getOrigin().z(),  Tmp.getOrigin().y()/Tmp.getOrigin().z(),
            Tpp.getOrigin().x()/Tpp.getOrigin().z(),  Tpp.getOrigin().y()/Tpp.getOrigin().z();
    std::cout<<" pointFeature in camera frame: "<< featurexy <<std::endl;

    VectorXs realZ;  //Z in cam frame
    realZ = VectorXs::Zero(4);
    realZ<< Tpm.getOrigin().z(), Tmm.getOrigin().z(),
            Tmp.getOrigin().z(), Tpp.getOrigin().z();
    std::cout<< " featureDepth: " << realZ << std::endl;

    jointStates << 0.79, 0.48, -0.0007, -0.754, 0.2, 0.508, 0.001;

    VectorXs stateX(19);   //initial state
    stateX.block(0,0,8,1) = featurexy;
    stateX.block(8,0,7,1) = jointStates;
    stateX.block(15,0,4,1) = realZ;

    const unsigned int nX = 19;  // 2*4 + 1*4 + 7
    const unsigned int nU = 7;   // 7 joints vel
    const unsigned int nY = 15;   // 2*4 + 7 joints position

    MatrixXs P(nX, nX), Qalpha(nX, nX), Qbeta(nU, nU), Qgamma(nY, nY);

    P = MatrixXs::Identity(nX, nX); //INITIAL
    P.block(0,0,8,19) = pp_xy * pp_xy * P.block(0,0,8,19);
    P.block(8,0,7,19) = pp_q * pp_q * P.block(8,0,7,19);
    P.block(15,0,4,19) = pp_Z * pp_Z * P.block(15,0,4,19);

    Qalpha = Qalpha.setIdentity();        //STATE
    Qalpha.block(0,0,8,19) = alpha_xy * alpha_xy * Qalpha.block(0,0,8,19);
    Qalpha.block(8,0,7,19) = alpha_q * alpha_q *Qalpha.block(8,0,7,19);
    Qalpha.block(15,0,4,19) = alpha_Z * alpha_Z * Qalpha.block(15,0,4,19);

    Qbeta  = beta_q * beta_q * Qbeta.setIdentity();         //INPUT

    Qgamma = Qgamma.setIdentity();        //MEASUREMENT
    Qgamma.block(0,0,8,15) = gama_xy * gama_xy * Qgamma.block(0,0,8,15);
    Qgamma.block(8,0,7,15) = gama_q * gama_q * Qgamma.block(8,0,7,15);

    ekf_task.initializeP(P);
    ekf_task.setQ_alpha(Qalpha);
    ekf_task.setQ_beta(Qbeta);
    ekf_task.setQ_gamma(Qgamma);
    ekf_task.initializeState(stateX);     //set noises

#ifdef VISP_HAVE_DISPLAY
    vpPlot plotter(4, 250 * 3, 750, 100, 200, "Real time curves plotter");
    plotter.setTitle(0, "real Z");
    plotter.setTitle(1, "camera velocities");
    plotter.setTitle(2, "Z error");
    plotter.setTitle(3, "xy error");

    plotter.initGraph(0, 4);
    plotter.initGraph(1, 6);
    plotter.initGraph(2, 4);
    plotter.initGraph(3, 8);

    plotter.setLegend(0, 0, "Z1");
    plotter.setLegend(0, 1, "Z2");
    plotter.setLegend(0, 2, "Z3");
    plotter.setLegend(0, 3, "Z4");


    plotter.setLegend(1, 0, "vx");
    plotter.setLegend(1, 1, "vy");
    plotter.setLegend(1, 2, "vz");
    plotter.setLegend(1, 3, "wx");
    plotter.setLegend(1, 4, "wy");
    plotter.setLegend(1, 5, "wz");

    plotter.setLegend(2, 0, "Z1");
    plotter.setLegend(2, 1, "Z2");
    plotter.setLegend(2, 2, "Z3");
    plotter.setLegend(2, 3, "Z4");

    plotter.setLegend(3, 0, "x1");
    plotter.setLegend(3, 1, "y1");
    plotter.setLegend(3, 2, "x2");
    plotter.setLegend(3, 3, "y2");
    plotter.setLegend(3, 4, "x3");
    plotter.setLegend(3, 5, "y3");
    plotter.setLegend(3, 6, "x4");
    plotter.setLegend(3, 7, "y4");
#endif     //plotter

    //set the desired feature
    VectorXs s_star(8);
    s_star << -0.08/Z_star, -0.08/Z_star, 0.08/Z_star, -0.08/Z_star, 0.08/Z_star, 0.08/Z_star, -0.08/Z_star, 0.08/Z_star;

//    MatrixXs sTn(6,6);
    MatrixXs sTn = MatrixXs::Identity(6,6);
    sTn(0,4) = -0.1;
    sTn(1,3) =  0.1;

    control_blocks_msgs::ArrayStamped jointVelCmd;

    VectorXs measurementVec(15), q_dot(7);
    std::vector<double> msg_vec;

    ekf_task.prediction(q_dot);

    MatrixXs Jac(6,7), interactionMat(8,6),  matTmp(8,7), camVel(6,1);

    //used to plot
    vpColVector vpErrZ(4), vpErrXY(8), vpvc(6), vpZ(4);

    double _x, _y, _Z;
    int iter = 0;
    ros::Rate loop_rate(1/dt);

    while (ros::ok()){

        listener.waitForTransform(camFrame,  "/boardbw_fix_circ_mm",  ros::Time(0), ros::Duration(3.0));
        listener.lookupTransform(camFrame,   "/boardbw_fix_circ_mm",  ros::Time(0), Tmm);

        listener.waitForTransform(camFrame,  "/boardbw_fix_circ_pm",  ros::Time(0), ros::Duration(3.0));
        listener.lookupTransform(camFrame,   "/boardbw_fix_circ_pm",  ros::Time(0), Tpm);

        listener.waitForTransform(camFrame,  "/boardbw_fix_circ_mp",  ros::Time(0), ros::Duration(3.0));
        listener.lookupTransform(camFrame,   "/boardbw_fix_circ_mp",  ros::Time(0), Tmp);

        listener.waitForTransform(camFrame,  "/boardbw_fix_circ_pp",  ros::Time(0), ros::Duration(3.0));
        listener.lookupTransform( camFrame,  "/boardbw_fix_circ_pp",  ros::Time(0), Tpp);

        //realZ in camera frame
        realZ<< Tpm.getOrigin().z(), Tmm.getOrigin().z(),
                Tmp.getOrigin().z(), Tpp.getOrigin().z();

        if (iter < itermax) {

            if (iter != 0){
                if(cameraFlag != 0)
                {
                    cameraAvaliable = true;
                    cameraFlag = 0;
                }else cameraAvaliable = false;

                if(encoderFlag != 0)
                {
                    encoderAvaliable = true;
                    encoderFlag = 0;
                }else encoderAvaliable = false;
            }

            if( cameraAvaliable && encoderAvaliable )
            {
//                std::cout<<"both sensor Avaliable" << std::endl;
                measurementVec.resize(15);
                measurementVec.block(0,0, 8, 1) = featurexy;  //featurexy comes from tracker in camera frame
                measurementVec.block(8, 0, 7, 1) = jointStates;
            }

//            else if( cameraAvaliable && !encoderAvaliable )
//            {
//                std::cout<<"only cameraAvaliable" << std::endl;
//                measurementVec.resize(8);
//                measurementVec = featurexy;
//            }
//            else if( !cameraAvaliable && encoderAvaliable )
//            {
//                std::cout<<"only encoderAvaliable" << std::endl;
//                measurementVec.resize(7);
//                measurementVec = jointStates;
//            }
//            else{
//                std::cout<<"not------------------ Avaliable" << std::endl;
//            }

            if(cameraAvaliable && encoderAvaliable)
            {
                ekf_task.setSensorMatrix(measurementVec);
                ekf_task.estimate(q_dot, measurementVec);
            }

            for (int i = 0; i<8; i++)
            {
                vpErrXY[i] =  featurexy(i) - stateX(i);
            }
            plotter.plot(3, iter, vpErrXY);

            stateX = ekf_task.getX();  //
            for( int i=0; i < 4; i++){
                _x = (1 - xyInLoop) * featurexy(2*i)     + xyInLoop*stateX(2*i);
                _y = (1 - xyInLoop) * featurexy(2*i + 1) + xyInLoop*stateX(2*i + 1);
                _Z = (1 - ZInLoop)  * realZ(i)           + ZInLoop*stateX(15+i);  //estimated or the realZ

                interactionMat(2 * i, 0) = -1.0 / _Z;
                interactionMat(2 * i, 1) = 0.0;
                interactionMat(2 * i, 2) = _x / _Z;
                interactionMat(2 * i, 3) = _x * _y;
                interactionMat(2 * i, 4) = -(1.0 + _x * _x);
                interactionMat(2 * i, 5) = _y;

                interactionMat(2 * i + 1, 0) = 0.0;
                interactionMat(2 * i + 1, 1) = -1.0 / _Z;
                interactionMat(2 * i + 1, 2) = _y / _Z;
                interactionMat(2 * i + 1, 3) = 1.0 + _y * _y;
                interactionMat(2 * i + 1, 4) = -_x * _y;
                interactionMat(2 * i + 1, 5) = -_x;
            }
            pinocchio::forwardKinematics(model, data, jointStates);
            pinocchio::frameJacobian(model, data, jointStates, model.getFrameId(endEffectorName), Jac);   //calculate JACOBIAN
            matTmp = interactionMat * sTn * Jac;
            featureError = featurexy - s_star;
            q_dot = -lamda * pseudoInverse(matTmp) * featureError;

//          std::cout<<" q_dot : " <<  q_dot << std::endl;

            msg_vec.clear();
            for(int i=0; i<7; i++)
            {
                msg_vec.push_back(q_dot(i));
            }

            jointVelCmd.header.frame_id = 7;
            jointVelCmd.header.seq = iter;
            jointVelCmd.header.stamp = ros::Time::now();

            jointVelCmd.data.clear();
            jointVelCmd.data.insert(jointVelCmd.data.end(), msg_vec.begin(), msg_vec.end());
            vel_cmd_pub.publish(jointVelCmd);

            ekf_task.prediction(q_dot);
            errorZ =  stateX.block(15,0,4,1) - realZ;

            for (int i = 0; i<4; i++) {
                vpZ[i] =  realZ(i);
            }

            camVel = ekf_task.getCamVel();
            for (int i = 0; i<6; i++) {
                vpvc[i] =  camVel(i);
            }
            for (int i = 0; i<4; i++) {
                vpErrZ[i] =  errorZ(i);
            }
            plotter.plot(0, iter, vpZ);
            plotter.plot(1, iter, vpvc);
            plotter.plot(2, iter, vpErrZ);
            iter++;
        }
        else break;

        ros::spinOnce();
        loop_rate.sleep();
    }
    return 0;
}