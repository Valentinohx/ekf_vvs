#ifndef __VS_cam_vel_input_hpp_
#define __VS_cam_vel_input_hpp_

#include "ekf_base.hpp"
#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <Eigen/Dense>


template<class Scalar>
class VS_cam_vel_input : public EKF<Scalar>
{
public:

    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
    typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVectorXs;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

    VS_cam_vel_input(int nPoints, Scalar dt);

    virtual void prediction(const Eigen::Ref<VectorXs>& Uk_);
    void getStateUpdated(const Eigen::Ref<VectorXs>& Uk_);

protected:
    int _nPoints;
    MatrixXs _interaction_Lxy;
    MatrixXs _interaction_LZ;
    Eigen::Matrix<Scalar, 6, 1> _camScrew;

private:
    Scalar _x;
    Scalar _y;
    Scalar _Z;

    MatrixXs _dl1_dsk;
    MatrixXs _dl2_dsk;
    MatrixXs _dlxy_dsk;

    MatrixXs _dl1_dzk;
    MatrixXs _dl2_dzk;
    MatrixXs _dlxy_dzk;

    MatrixXs _dlz_dsk;
    MatrixXs _dlz_dzk;

    MatrixXs _dh1_dsk;
    MatrixXs _dh2_dsk;

    MatrixXs _dh1_dzk;
    MatrixXs _dh2_dzk;

    MatrixXs _dh1_duk;
    MatrixXs _dh2_duk;

    void calcuInteraction_LxyZ();

    //calculate Kalman Jacobian
    bool calcuAk(const Eigen::Ref<VectorXs>& Uk_);   // dh/dx
    bool calcuBk();   // dh/du

    //note:
    //since dg/du is a constant matrix in this case, so just set it once
    // in the constructor instead of in each iteration
};
#include "ekf_vvs/VS_cam_vel_input.hxx"

#endif
