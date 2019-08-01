#ifndef __EKF_base_hpp__
#define __EKF_base_hpp__

#include <Eigen/Dense>
#include <iostream>


template<class Scalar >
class EKF
{
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

    //constructor to set all the matrix size
    EKF(int x_dim, int u_dim, int y_dim, Scalar dt_);

    //used to set the _Ak, _Bk, _Ck and evolution model
    virtual void prediction( const Eigen::Ref<VectorXs>& Uk_ ) = 0;

    //to update the covariance matrix, the estimate based on measurements
    void estimate( const Eigen::Ref<VectorXs>& Uk_, const Eigen::Ref<VectorXs>& Yk_ );

    //to calculate the kalman gain Kk by using _InverseSolver
    void calcuKalmanGain();

    //setters and initializers
    //initilizer
    void initializeState( const Eigen::Ref<VectorXs>& stateInitial_);
    void initializeP( const Eigen::Ref<MatrixXs>& initialUncertainty);

    void setSampleTime(const Scalar& dt);
    Scalar getSampleTime();

    //setters
    void setQ_alpha(const Eigen::Ref<MatrixXs>& Q_alpha_);
    void setQ_beta( const Eigen::Ref<MatrixXs>& Q_beta_);
    virtual void setQ_gamma( const Eigen::Ref<MatrixXs>& Q_gama_);

    //getter
    VectorXs& getX(){ return this->_Xk;}

protected:

    VectorXs _Xk;   // state vector
    VectorXs _Uk;   // current input
    VectorXs _Yk;   // measurement from sensor
    VectorXs _gXk;  // measurement equation


    MatrixXs _Qalpha;  // state noise
    MatrixXs _Qbeta;   // input noise
    MatrixXs _Qgama;   // Measurement noise

    MatrixXs _Pk;      //propagation covariance
    MatrixXs _Kk;      //kalman gain

    //Kalman Jacobian,
    // x_k+1/k = f(x_k/k, U_k) evolution model
    // y_k = g(x_k+1/k)     measurement equation

    MatrixXs _Ak;       //df/dx
    MatrixXs _Bk;       //df/du
    MatrixXs _Ck;       //dg/dx

    int _x_dim_n;       // state dim
    int _u_dim_m;       // input dim
    int _y_dim_p;       // measurement dim

    Scalar _dt;         //sampling time

    MatrixXs _innovationS;  //innocation
    MatrixXs _Sinverse;     //inverse of innocation
    Eigen::PartialPivLU<MatrixXs> _InverseSolver;   //solver used to speed up matrix inverse

    //checkers used to check if the kalman filter has been initialized correctly
    bool isPinitialized = false;
    bool isXinitialized = false;
    bool isQalphaSet    = false;
    bool isQbetaSet     = false;
    bool isQgammaSet    = false;
};

#include "ekf_vvs/ekf_base.hxx"

#endif
