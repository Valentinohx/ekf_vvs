#ifndef __VS_with_robot_joints_hxx_
#define __VS_with_robot_joints_hxx_

#include <iostream>
#include <exception>

template<class Scalar>
VS_with_robot_joints<Scalar>::VS_with_robot_joints(
        int nPoints,
        Scalar dt,
        pinocchio::Model model_
)
        : EKF<Scalar>(3*nPoints + model_.nq, model_.nq, 2*nPoints + model_.nq, dt),
          _model(model_),
          _data(_model)
{
    _nPoints = nPoints;
    _nJoints = _model.nv;

    _JntAcc = VectorXs::Zero(_model.nv);
    _Jac = MatrixXs::Zero(6, _model.nv);

    sTn = MatrixXs::Identity(6,6);
    sTn(0,4) = 0.1;
    sTn(1,3) = -0.1;

    _qk.data =  VectorXs::Zero(_nJoints);

    _interaction_LZ = MatrixXs::Zero(_nPoints, 6);
    _interaction_Lxy = MatrixXs::Zero(2*_nPoints, 6);

    _dl1_dsk =  MatrixXs::Zero(6,2);
    _dl2_dsk =  MatrixXs::Zero(6,2);

    _dl1_dzk =  MatrixXs::Zero(6,1);
    _dl2_dzk =  MatrixXs::Zero(6,1);

    _dlz_dsk =  MatrixXs::Zero(6,2);
    _dlz_dzk =  MatrixXs::Zero(6,1);

    _dh1_dsk =  MatrixXs::Zero(2*_nPoints, 2*_nPoints); //8*8
    _dh2_dsk =  MatrixXs::Zero(_nJoints, 2*_nPoints);   //7*8
    _dh3_dsk =  MatrixXs::Zero(_nPoints, 2*_nPoints);   //4*8

    _dh1_dqk =  MatrixXs::Zero(2*_nPoints, _nJoints);   //8*7
    _dh2_dqk =  MatrixXs::Zero(_nJoints, _nJoints);     //7*7
    _dh3_dqk =  MatrixXs::Zero(_nPoints, _nJoints);     //4*7

    _dh1_dzk =  MatrixXs::Zero(2*_nPoints, _nPoints);   //8*4
    _dh2_dzk =  MatrixXs::Zero(_nJoints, _nPoints);     //7*4
    _dh3_dzk =  MatrixXs::Zero(_nPoints, _nPoints);     //4*4

    _dlxy_dsk =  MatrixXs::Zero(2, 2);                  //2*2
    _dlxy_dzk =  MatrixXs::Zero(2, 1);                  //2*1

    _dh1_dqk_dot =  MatrixXs::Zero(2*_nPoints, _nJoints);   //8*7
    _dh2_dqk_dot =  MatrixXs::Zero(_nJoints, _nJoints);     //7*7
    _dh3_dqk_dot =  MatrixXs::Zero(_nPoints, _nJoints);     //4*7

    _dvdq = MatrixXs::Zero(6, _model.nv);
    this->_Ak.setZero();
    this->_Bk.setZero();
    this->_Ck.setZero();

    (this->_Ck.block(0, 0, 2*_nPoints, 2*_nPoints)).setIdentity();
    (this->_Ck.block(2*_nPoints, 2*_nPoints, _nJoints, _nJoints)).setIdentity();
}

template<class Scalar>
void VS_with_robot_joints<Scalar>::calcuInteraction_LxyZ()
{
    for(int i = 0; i < _nPoints; i++)
    {
        _x = this->_Xk(2*i);
        _y = this->_Xk(2*i + 1);
        _Z = this->_Xk(2*_nPoints + _nJoints + i);
        if (_Z > 0)
        {
            _interaction_Lxy(2*i, 0) = -1.0 / _Z;
            _interaction_Lxy(2*i, 2) =  _x / _Z;
            _interaction_Lxy(2*i, 3) =  _x * _y;
            _interaction_Lxy(2*i, 4) = -( 1.0 + _x * _x );
            _interaction_Lxy(2*i, 5) =  _y;

            _interaction_Lxy(2*i + 1, 1) = -1.0 / _Z;
            _interaction_Lxy(2*i + 1, 2) = _y / _Z;
            _interaction_Lxy(2*i + 1, 3) = 1.0 + _y * _y;
            _interaction_Lxy(2*i + 1, 4) = -_x * _y;
            _interaction_Lxy(2*i + 1, 5) = -_x;

            _interaction_LZ(i, 2) = -1.0;
            _interaction_LZ(i, 3) = -_y * _Z;
            _interaction_LZ(i, 4) =  _x * _Z;
        }
        else
        {
            throw std::runtime_error("VS_with_robot_joints::calcuInteraction_LxyZ(): invalid depth!");
        }
    }
}

template<class Scalar>
void VS_with_robot_joints<Scalar>::calcuDvDq_Jac( const Eigen::Ref<VectorXs>& Uk_ )
{
    _qk.data = this->_Xk.block(2*_nPoints, 0, _nJoints, 1);
    pinocchio::computeForwardKinematicsDerivatives(_model, _data, _qk.data,  Uk_, _JntAcc);
    pinocchio::Model::JointIndex ID = _model.getJointId("fkuka_lwr4p_a6_joint");
    pinocchio::getJointVelocityDerivatives(_model, _data, ID, pinocchio::ReferenceFrame::LOCAL, _dvdq, _Jac);
    _dvdq = sTn * _dvdq;   // dcVdq = cTe * deVdq
    _Jac  = sTn * _Jac;    // cJ = cTe * eJ;
}

template<class Scalar>
void VS_with_robot_joints<Scalar>::calcuCamScrew(const Eigen::Ref<VectorXs>& Uk_)
{
    _camScrew = _Jac * Uk_;   //cV = cJ * q_dot

}

template<class Scalar>
void VS_with_robot_joints<Scalar>::getStateUpdated(const Eigen::Ref<VectorXs>& Uk_)
{
    this->_Xk.block(0, 0, 2*_nPoints, 1) +=  _interaction_Lxy * _camScrew * this->_dt;
    this->_Xk.block(2*_nPoints, 0, _nJoints, 1) +=  Uk_ * this->_dt;
    this->_Xk.block(2*_nPoints+_nJoints, 0, _nPoints, 1) += _interaction_LZ * _camScrew * this->_dt;

    this->_gXk =  this->_Xk.block(0, 0, 2*_nPoints + _nJoints, 1);

//    std::cout<<"VS_with_robot_joints::getStateUpdated called" << std::endl;
}

// dh/dx
template<class Scalar>
void VS_with_robot_joints<Scalar>::calcuAk ( const Eigen::Ref<VectorXs>& Uk_ )
{
    for(int i = 0; i < _nPoints; i++){
        _x = this->_Xk(2*i);
        _y = this->_Xk(2*i + 1);
        _Z = this->_Xk(2*_nPoints+_nJoints + i);
        if (_Z > 0){
            _dl1_dsk(2, 0) = 1.0 / _Z;
            _dl1_dsk(3, 0) = _y;
            _dl1_dsk(3, 1) = _x;
            _dl1_dsk(4, 0) = -2.0 * _x;
            _dl1_dsk(5, 1) = 1.0;

            _dl2_dsk(2, 1) = 1.0 / _Z;
            _dl2_dsk(3, 1) = 2.0 * _y;
            _dl2_dsk(4, 0) = -_y;
            _dl2_dsk(4, 1) = -_x;
            _dl2_dsk(5, 0) = -1.0;  // <#ok>

            _dlxy_dsk.row(0) = _camScrew.transpose() * _dl1_dsk;
            _dlxy_dsk.row(1) = _camScrew.transpose() * _dl2_dsk;
            _dh1_dsk.block(2*i, 2*i, 2, 2) = MatrixXs::Identity(2, 2) + this->_dt*_dlxy_dsk; // <#ok>

            _dl1_dzk(0, 0) =  1.0  / ( _Z * _Z );
            _dl1_dzk(2, 0) = -_x / ( _Z * _Z );
            _dl2_dzk(1, 0) =  1.0  / ( _Z * _Z );
            _dl2_dzk(2, 0) = -_y / ( _Z * _Z );  // <#ok>

            _dlxy_dzk.row(0) = _camScrew.transpose() * _dl1_dzk;
            _dlxy_dzk.row(1) = _camScrew.transpose() * _dl2_dzk;
            _dh1_dzk.block(2*i, i, 2, 1) =  this->_dt*_dlxy_dzk; // <#ok>

            _dlz_dsk(3, 1) = -_Z;
            _dlz_dsk(4, 0) =  _Z;
            _dh3_dsk.block(i, 2*i, 1, 2) = _camScrew.transpose() * _dlz_dsk *  this->_dt;   // <#ok>

            _dlz_dzk(3, 0) = -_y;
            _dlz_dzk(4, 0) =  _x;
            _dh3_dzk.block(i, i, 1, 1) = MatrixXs::Identity(1, 1) + this->_dt*_camScrew.transpose()*_dlz_dzk; // <#ok>   //check
        }
        else{
            throw std::runtime_error("VS_with_robot_joints::calcuAk(): invalid depth!");
        }
    }
    //            this->_Ak <<_dh1_dsk, _dh1_dqk, _dh1_dzk,
//                    _dh2_dsk, _dh2_dqk, _dh2_dzk,
//                    _dh3_dsk, _dh3_dqk, _dh3_dzk;
    _dh2_dsk.setZero();
    _dh2_dqk.setIdentity();
    _dh2_dzk.setZero();     // <#ok>

    _dh1_dqk = _interaction_Lxy * _dvdq * this->_dt;  //6*8 * 6*7
    _dh3_dqk = _interaction_LZ  * _dvdq * this->_dt;

    this->_Ak.block(0,0,8,8) = _dh1_dsk;  //8*8
    this->_Ak.block(0,8,8,7) = _dh1_dqk;  //8*7
    this->_Ak.block(0,15,8,4) = _dh1_dzk;  //8*4

    this->_Ak.block(8,0,7,8) = _dh2_dsk;   //7*8
    this->_Ak.block(8,8,7,7) = _dh2_dqk;   //7*7
    this->_Ak.block(8,15,7,4) = _dh2_dzk;   //7*4

    this->_Ak.block(15,0,4,8) = _dh3_dsk;   //4*8
    this->_Ak.block(15,8,4,7) = _dh3_dqk;   //4*7
    this->_Ak.block(15,15,4,4) = _dh3_dzk;   //4*4
}

template<class Scalar>
void VS_with_robot_joints<Scalar>::calcuBk ()
{
    _dh1_dqk_dot = _interaction_Lxy * _Jac * this->_dt;
    _dh2_dqk_dot =  MatrixXs::Identity(_nJoints, _nJoints) * this->_dt;
    _dh3_dqk_dot = _interaction_LZ * _Jac * this->_dt;

    this->_Bk << _dh1_dqk_dot, _dh2_dqk_dot, _dh3_dqk_dot;
}

template<class Scalar>
void VS_with_robot_joints<Scalar>::prediction(const Eigen::Ref<VectorXs>& Uk_)
{
    calcuDvDq_Jac(Uk_);
    calcuCamScrew (Uk_);
    calcuInteraction_LxyZ();

    this->calcuAk(Uk_);
    this->calcuBk();
    this->_Pk  =  this->_Ak * this->_Pk * this->_Ak.transpose() + this->_Bk * this->_Qbeta * this->_Bk.transpose() + this->_Qalpha;

    getStateUpdated(Uk_);
}

#endif
