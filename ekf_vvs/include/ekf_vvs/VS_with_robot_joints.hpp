
#ifndef __VS_with_robot_joints_hpp__
#define __VS_with_robot_joints_hpp__

#include "ekf_base.hpp"

#include <Eigen/Dense>

#include <pinocchio/multibody/model.hpp>
#include <pinocchio/parsers/urdf.hpp>
#include <pinocchio/algorithm/joint-configuration.hpp>
#include <pinocchio/algorithm/kinematics.hpp>
#include <pinocchio/algorithm/kinematics-derivatives.hpp>
#include <pinocchio/algorithm/frames.hpp>
#include <pinocchio/algorithm/jacobian.hpp>

#include <kdl/chain.hpp>
#include <kdl/chainfksolver.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl_parser/kdl_parser.hpp>
#include <kdl/treefksolverpos_recursive.hpp>
#include <kdl/treejnttojacsolver.hpp>
#include <kdl/chainfksolverpos_recursive.hpp>
#include <kdl/chainjnttojacsolver.hpp>
#include <kdl/chainjnttojacdotsolver.hpp>
#include <kdl/frames_io.hpp>

template<class Scalar>
class VS_with_robot_joints : public EKF<Scalar>
{
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
    typedef Eigen::Matrix<Scalar, 1, Eigen::Dynamic> RowVectorXs;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

    VS_with_robot_joints(
            int nPoints,
            Scalar dt,
            pinocchio::Model model_
    );

    void prediction(const Eigen::Ref<VectorXs>& Uk_);
    virtual void getStateUpdated(const Eigen::Ref<VectorXs>& Uk_);

    MatrixXs& getMatInteraction(){ return this->_interaction_Lxy;}
    MatrixXs getCamVel(){ return this->_camScrew;}
    MatrixXs& getJac(){ return this->_Jac;}

    void calcuInteraction_LxyZ();
    void calcuDvDq_Jac( const Eigen::Ref<VectorXs>& Uk_ );

protected:
    int _nPoints;
    int _nJoints;
    MatrixXs _interaction_Lxy;
    MatrixXs _interaction_LZ;
    Eigen::Matrix<Scalar, 6, 1> _camScrew;

    KDL::JntArray _qk;
    MatrixXs sTn;      //camera has a rotation wrt. the last joint of the robot

    pinocchio::Model _model;
    pinocchio::Data  _data;
    MatrixXs _Jac;
    VectorXs _JntAcc;

private:
    Scalar _x;
    Scalar _y;
    Scalar _Z;

    MatrixXs _LxyZ;

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
    MatrixXs _dh3_dsk;

    MatrixXs _dh1_dqk;
    MatrixXs _dh2_dqk;
    MatrixXs _dh3_dqk;

    MatrixXs _dh1_dzk;
    MatrixXs _dh2_dzk;
    MatrixXs _dh3_dzk;

    MatrixXs _dh1_dqk_dot;
    MatrixXs _dh2_dqk_dot;
    MatrixXs _dh3_dqk_dot;

    MatrixXs _dvdq;


    void calcuCamScrew (const Eigen::Ref<VectorXs>& Uk_);

    //calculate Filter Jacobian
    void calcuAk(const Eigen::Ref<VectorXs>& Uk_);  // dh/dx
    void calcuBk( );   // dh/du
};

#include "ekf_vvs/VS_with_robot_joints.hxx"

#endif
