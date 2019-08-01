
#include <visp3/robot/vpSimulatorCamera.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/vs/vpServo.h>

#include "ekf_vvs/ekf_base.hpp"
#include <ekf_vvs/VS_cam_vel_input.hpp>

#include <iostream>
#include <Eigen/Dense>
#include <kdl_parser/kdl_parser.hpp>
#include <ros/ros.h>

#include <ctime>
#include <ratio>
#include <chrono>

#include <visp3/gui/vpPlot.h>
#include <visp3/robot/vpSimulatorCamera.h>
#include <visp3/visual_features/vpFeatureBuilder.h>
#include <visp3/vs/vpServo.h>
#include <visp3/core/vpImageConvert.h>
#include <visp3/io/vpImageIo.h>


int main(int argc, char** argv)
{
    typedef Eigen::Matrix<double, Eigen::Dynamic, 1> VectorXs;
    typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

    const unsigned int nX = 12;
    const unsigned int nU = 6;
    const unsigned int nY = 8;
    const unsigned int nP = 4;
    const unsigned int nJ = 0;
    const float dt = 0.040;

    VS_cam_vel_input<double> vs(nP, dt);
    //initialize P, Qalpha, Qbeta, Qgamma;
    
    MatrixXs P, Qalpha, Qbeta, Qgamma;
    
    P.resize(nX, nX);
    Qalpha.resize(nX, nX);
    Qbeta.resize(nU, nU);
    Qgamma.resize(nY, nY);
    
    P      = 0.001 * MatrixXs::Identity(nX, nX);       //INITIAL
    Qalpha = 0.01 * Qalpha.setIdentity();  //STATE
    Qbeta  = 0.001 * Qbeta.setIdentity();  //INPUT
//    Qbeta  = 0.1 * Qbeta.setIdentity();  //INPUT
    Qgamma = 0.001 * Qgamma.setIdentity();  //MEASUREMENT

    vs.initializeP(P);
    vs.setQ_alpha(Qalpha);
    vs.setQ_beta(Qbeta);
    vs.setQ_gamma(Qgamma);
    
    vpColVector v(6);
    Eigen::Map<VectorXs> Uk(v.data, v.size());

    vpColVector errZ(4);
    vpColVector errxy(8);

    try {
        vpHomogeneousMatrix cdMo(0, 0, 0.75, 0, 0, 0);
        vpHomogeneousMatrix cMo(0.15, -0.1, 1., vpMath::rad(10), vpMath::rad(-10), vpMath::rad(50));
        std::cout<<" cMo: " << cMo << std::endl;
        vpPoint point[4];
        point[0].setWorldCoordinates(-0.1, -0.1, 0);
        point[1].setWorldCoordinates(0.1, -0.1, 0);
        point[2].setWorldCoordinates(0.1, 0.1, 0);
        point[3].setWorldCoordinates(-0.1, 0.1, 0);
        vpServo task;
        task.setServo(vpServo::EYEINHAND_CAMERA);
        task.setInteractionMatrixType(vpServo::CURRENT);
        task.setLambda(0.5);
//        task.setLambda(0.001);

        vpFeaturePoint p[4], pd[4];
        double featureVec[12];

        for (unsigned int i = 0; i < 4; i++)
        {
            point[i].track(cdMo);
            vpFeatureBuilder::create(pd[i], point[i]);
            point[i].track(cMo);
            vpFeatureBuilder::create(p[i], point[i]);

            featureVec[2*i]   = p[i].get_x();
            featureVec[2*i+1] = p[i].get_y();
            featureVec[8 + i] = p[i].get_Z();

            task.addFeature(p[i], pd[i]);
        }
        Eigen::Map<VectorXs> x_init(featureVec, 12);

        vs.initializeState(x_init);

        vpHomogeneousMatrix wMc, wMo;
        vpSimulatorCamera robot;
        robot.setSamplingTime(dt);
        robot.getPosition(wMc);
        wMo = wMc * cMo;

#ifdef VISP_HAVE_DISPLAY
        vpPlot plotter(4, 800, 800, 100, 200, "Real time curves plotter");
        plotter.setTitle(0, "Visual features error: S-S*");
        plotter.setTitle(1, "Camera velocities: v_c");
        plotter.setTitle(2, "Feature depth error: Z_est - Z_real");
        plotter.setTitle(3, "Feature coordinates error: S_est - S_real");

        plotter.initGraph(0, 8);
        plotter.initGraph(1, 6);
        plotter.initGraph(2, 4);
        plotter.initGraph(3, 8);

        plotter.setLegend(0, 0, "x1");
        plotter.setLegend(0, 1, "y1");
        plotter.setLegend(0, 2, "x2");
        plotter.setLegend(0, 3, "y2");
        plotter.setLegend(0, 4, "x3");
        plotter.setLegend(0, 5, "y3");
        plotter.setLegend(0, 6, "x4");
        plotter.setLegend(0, 7, "y4");
        plotter.setUnitX(0, "iteration N" );

        plotter.setLegend(1, 0, "vx");
        plotter.setLegend(1, 1, "vy");
        plotter.setLegend(1, 2, "vz");
        plotter.setLegend(1, 3, "wx");
        plotter.setLegend(1, 4, "wy");
        plotter.setLegend(1, 5, "wz");
        plotter.setUnitX(1, "iteration N" );
        plotter.setUnitY(1, "m/s;rad/s");


        plotter.setLegend(2, 0, "e_Z1");
        plotter.setLegend(2, 1, "e_Z2");
        plotter.setLegend(2, 2, "e_Z3");
        plotter.setLegend(2, 3, "e_Z4");
        plotter.setUnitX(2, "iteration N" );
        plotter.setUnitY(2, "m" );



        plotter.setLegend(3, 0, "e_x1");
        plotter.setLegend(3, 1, "e_y1");
        plotter.setLegend(3, 2, "e_x2");
        plotter.setLegend(3, 3, "e_y2");
        plotter.setLegend(3, 4, "e_x3");
        plotter.setLegend(3, 5, "e_y3");
        plotter.setLegend(3, 6, "e_x4");
        plotter.setLegend(3, 7, "e_y4");
        plotter.setUnitX(3, "iteration N" );

#endif
        double measurementVec[8];

        //iteration
        for (unsigned int iter = 0; iter < 200; iter++)
        {
            robot.getPosition(wMc);
            cMo = wMc.inverse() * wMo;

            for (unsigned int i = 0; i < 4; i++) {
                point[i].track(cMo);
                vpFeatureBuilder::create(p[i], point[i]);
//                p[i].set_Z(0.75);

                measurementVec[2*i]   = p[i].get_x();
                measurementVec[2*i+1] = p[i].get_y();
            }
            v = task.computeControlLaw();
            robot.setVelocity(vpRobot::CAMERA_FRAME, v);


            robot.getPosition(wMc);
            cMo = wMc.inverse() * wMo;
            for (unsigned int i = 0; i < 4; i++) {
                point[i].track(cMo);
                vpFeatureBuilder::create(p[i], point[i]);
                measurementVec[2*i]   = p[i].get_x();
                measurementVec[2*i+1] = p[i].get_y();
//                p[i].set_Z(0.75);
            }
            Eigen::Map<VectorXs> Ymeasurement(measurementVec, 8);

            vs.prediction( Uk );
            vs.estimate(Uk, Ymeasurement);

            VectorXs X = vs.getX();
            double z0 = p[0].get_Z();
            errZ[0] = X(8,0) -  p[0].get_Z();
            errZ[1] = X(9,0) -  p[1].get_Z();
            errZ[2] = X(10,0) - p[2].get_Z();
            errZ[3] = X(11,0) - p[3].get_Z();

            errxy[0] = X(0,0) -  p[0].get_x();
            errxy[1] = X(1,0) -  p[0].get_y();
            errxy[2] = X(2,0) -  p[1].get_x();
            errxy[3] = X(3,0) -  p[1].get_y();
            errxy[4] = X(4,0) -  p[2].get_x();
            errxy[5] = X(5,0) -  p[2].get_y();
            errxy[6] = X(6,0) -  p[3].get_x();
            errxy[7] = X(7,0) -  p[3].get_y();

#ifdef VISP_HAVE_DISPLAY
            plotter.plot(0, iter, task.getError());
            plotter.plot(1, iter, v);
            plotter.plot(2, iter, errZ);
            plotter.plot(3, iter, errxy);
#endif
        }
        task.kill();

#ifdef VISP_HAVE_DISPLAY
//        plotter.saveData(0, "/home/valentinohx/Dropbox/report/experimentsFigures/err.png");
        vpImageIo::write(plotter.I, "/home/valentinohx/Dropbox/report/experimentsFigures/err.png"); // Gray
        plotter.saveData(1, "/home/valentinohx/Dropbox/report/experimentsFigures/vc.dat");
        plotter.saveData(2, "/home/valentinohx/Dropbox/report/experimentsFigures/errZ.dat");
        vpDisplay::getClick(plotter.I);
#endif

    } catch (vpException &e) {
        std::cout << "Catch an exception: " << e << std::endl;
    }
}
