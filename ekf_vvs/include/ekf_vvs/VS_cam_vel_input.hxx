#ifndef __VS_cam_vel_input_hxx_
#define __VS_cam_vel_input_hxx_

#include <iostream>
#include <stdlib.h>     /* exit, EXIT_FAILURE */

template<class Scalar>
VS_cam_vel_input<Scalar>::VS_cam_vel_input(
        int nPoints,
        Scalar dt
        )
    : EKF<Scalar>(3*nPoints, 6, 2*nPoints, dt)
{
    _nPoints = nPoints;
    _interaction_Lxy =  MatrixXs::Zero(2*_nPoints, 6);
    _interaction_LZ =  MatrixXs::Zero(_nPoints, 6);

    _dl1_dsk =  MatrixXs::Zero(6,2);
    _dl2_dsk =  MatrixXs::Zero(6,2);

    _dl1_dzk =  MatrixXs::Zero(6,1);
    _dl2_dzk =  MatrixXs::Zero(6,1);

    _dlz_dsk =  MatrixXs::Zero(6,2);
    _dlz_dzk =  MatrixXs::Zero(6,1);

    _dh1_dsk =  MatrixXs::Zero(2*_nPoints, 2*_nPoints); //8*8
    _dh2_dsk =  MatrixXs::Zero(_nPoints, 2*_nPoints);   //4*8

    _dh1_dzk =  MatrixXs::Zero(2*_nPoints, _nPoints);   //8*4

    _dh2_dzk =  MatrixXs::Zero(_nPoints, _nPoints);     //4*4

    _dlxy_dsk =  MatrixXs::Zero(2, 2);                  //2*2
    _dlxy_dzk =  MatrixXs::Zero(2, 1);                  //2*1

    _dh1_duk =  MatrixXs::Zero(2, 6);
    _dh2_duk =  MatrixXs::Zero(1, 6);

    this->_Ak.setZero();
    this->_Bk.setZero();
    this->_Ck.setZero();

    // dg/du calculated here since in this special case it is just
    // a constant matrix
    (this->_Ck.block(0, 0, 2*_nPoints, 2*_nPoints)).setIdentity();
}


template<class Scalar>
void VS_cam_vel_input<Scalar>::calcuInteraction_LxyZ()
{
    for(int i = 0; i < _nPoints; i++)
    {
        _x = this->_Xk(2*i);
        _y = this->_Xk(2*i + 1);
        _Z = this->_Xk(2*_nPoints + i);

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
            throw std::runtime_error("VS_cam_vel_input::calcuInteraction_LxyZ(): invalid depth!");
        }
    }
}

template<class Scalar>
void VS_cam_vel_input<Scalar>::getStateUpdated(const Eigen::Ref<VectorXs>& Uk_)
{
    if ( this->isXinitialized && this->isPinitialized && this->isQalphaSet &&
         this->isQbetaSet && this->isQgammaSet)
    {
        calcuInteraction_LxyZ();

        this->_Xk.block(0, 0, 2*_nPoints, 1) += _interaction_Lxy * Uk_ * this->_dt;
        this->_gXk =  this->_Xk.block(0, 0, 2*_nPoints, 1);
        this-> _Xk.block(2*_nPoints, 0, _nPoints, 1) +=  _interaction_LZ * Uk_ * this->_dt;
    }
    else
    {
        throw std::runtime_error("VS_cam_vel_input::getStqteUpdate(): initialize needed!");
    }
}

// dh/dx
template<class Scalar>
bool VS_cam_vel_input<Scalar>::calcuAk (const Eigen::Ref<VectorXs>& Uk_)
{

    for(int i = 0; i < _nPoints; i++)
    {
        _x = this->_Xk(2*i);
        _y = this->_Xk(2*i + 1);
        _Z = this->_Xk(2*_nPoints + i);


        if (_Z > 0)
        {
            //            _dl1_dsk<< 0,     0,
            //                       0,     0,
            //                       1/_Z,  0,
            //                       _y,   _x,
            //                       -2*_x, 0,
            //                       0,     1;

            //            _dl2_dsk<< 0,    0,
            //                       0,    0,
            //                       0,   1/_Z,
            //                       0,   2*_y,
            //                       -_y, -_x,
            //                       -1,   0;

            _dl1_dsk(2, 0) = 1.0 / _Z;
            _dl1_dsk(3, 0) = _y;
            _dl1_dsk(3, 1) = _x;
            _dl1_dsk(4, 0) = -2.0 * _x;
            _dl1_dsk(5, 1) = 1.0;

            _dl2_dsk(2, 1) = 1.0 / _Z;
            _dl2_dsk(3, 1) = 2.0 * _y;
            _dl2_dsk(4, 0) = -_y;
            _dl2_dsk(4, 1) = -_x;
            _dl2_dsk(5, 0) = -1.0;

            _dlxy_dsk.row(0) = Uk_.transpose() * _dl1_dsk;
            _dlxy_dsk.row(1) = Uk_.transpose() * _dl2_dsk;
            _dh1_dsk.block(2*i, 2*i, 2, 2) = MatrixXs::Identity(2, 2) + _dlxy_dsk * this->_dt;

            //            _dl1_dzk<< 1/(_Z*_Z), 0,  -_x/(_Z*_Z), 0, 0, 0;
            //            _dl2_dzk<< 0, 1/(_Z*_Z),  -_y/(_Z*_Z), 0, 0, 0;

            _dl1_dzk(0, 0) =  1.0  / ( _Z * _Z );
            _dl1_dzk(2, 0) = -_x / ( _Z * _Z );

            _dl2_dzk(1, 0) =  1.0  / ( _Z * _Z );
            _dl2_dzk(2, 0) = -_y / ( _Z * _Z );

            _dlxy_dzk.row(0) = Uk_.transpose() * _dl1_dzk;
            _dlxy_dzk.row(1) = Uk_.transpose() * _dl2_dzk;
            _dh1_dzk.block(2*i, i, 2, 1) =  this->_dt * _dlxy_dzk;

            //            _dlz_dsk << 0,0,
            //                        0,0,
            //                        0,0,
            //                        0,-_Z,
            //                        _Z, 0,
            //                        0, 0;
            _dlz_dsk(3, 1) = -_Z;
            _dlz_dsk(4, 0) =  _Z;

            _dh2_dsk.block(i, 2*i, 1, 2) =  Uk_.transpose()*_dlz_dsk * this->_dt;

            //            _dlz_dzk<< 0, 0, 0, -_y, _x, 0;
            _dlz_dzk(3, 0) = -_y;
            _dlz_dzk(4, 0) =  _x;
            _dh2_dzk.block(i, i, 1, 1) = MatrixXs::Identity(1, 1) + Uk_.transpose()*_dlz_dzk * this->_dt;
        }
        else
        {
            throw std::runtime_error("VS_cam_vel_input::calcuAk(): invalid depth!");
            return false;
        }
    }
    this->_Ak <<_dh1_dsk,  _dh1_dzk,
            _dh2_dsk,  _dh2_dzk;
    return true;
}

template<class Scalar>
bool VS_cam_vel_input<Scalar>::calcuBk ()
{
    _dh1_duk = this->_dt * _interaction_Lxy;
    _dh2_duk = this->_dt * _interaction_LZ;

    this->_Bk << _dh1_duk, _dh2_duk;
    return true;
}

template<class Scalar>
void VS_cam_vel_input<Scalar>::prediction(const Eigen::Ref<VectorXs>& Uk_)
{
    this->calcuAk (Uk_);
    this->calcuBk ();

    this->_Pk  =  this->_Ak * this->_Pk * this->_Ak.transpose() + this->_Bk * this->_Qbeta * this->_Bk.transpose() + this->_Qalpha;

    this->getStateUpdated(Uk_);
}

#endif
