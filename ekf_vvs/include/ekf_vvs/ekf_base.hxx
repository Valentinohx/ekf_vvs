#ifndef __ekf_base_hxx__
#define __ekf_base_hxx__

template<class Scalar>
EKF<Scalar>::EKF(int x_dim, int u_dim, int y_dim, Scalar dt_)
{
    _x_dim_n = x_dim;
    _u_dim_m = u_dim;
    _y_dim_p = y_dim;
    _dt = dt_;

    _Xk.resize(_x_dim_n, 1);
    _Uk.resize(_u_dim_m, 1);

    _Qalpha.resize(_x_dim_n, _x_dim_n);
    _Qbeta.resize(_u_dim_m, _u_dim_m);
    _Pk.resize(_x_dim_n, _x_dim_n);
    _Ak.resize(_x_dim_n, _x_dim_n);
    _Bk.resize(_x_dim_n, _u_dim_m);

    _Qgama.resize(_y_dim_p, _y_dim_p);
    _Kk.resize(_x_dim_n, _y_dim_p);
    _Yk.resize(_y_dim_p, 1);
    _gXk.resize(_y_dim_p, 1);
    _Ck.resize(_y_dim_p, _x_dim_n);
    _innovationS.resize(_y_dim_p, _y_dim_p);
}

template<class Scalar >
void EKF<Scalar>::setSampleTime(const Scalar& dt)
{
    _dt = dt;
}

template<class Scalar >
Scalar EKF<Scalar>::getSampleTime()
{
    return _dt;
}

template<class Scalar > void EKF<Scalar>::estimate( const Eigen::Ref<VectorXs>& Uk_, const Eigen::Ref<VectorXs>& Yk_ )
{
    //Ak -> Bk -> X_k+1 -> Ck

//    prediction(Uk_);    //
    // P_k+1/k
//    _Pk  =  _Ak * _Pk * _Ak.transpose() + _Bk * _Qbeta * _Bk.transpose() + _Qalpha;

    //Kk
    calcuKalmanGain();

    //    //    //X_k+1/k+1
    _Xk  += _Kk*( Yk_ - _gXk);

    //P_k+1/k+1
    _Pk  =  ( MatrixXs::Identity(_x_dim_n, _x_dim_n) - _Kk * _Ck ) * _Pk;
}


template<class Scalar>
void EKF<Scalar>::calcuKalmanGain()
{
    _innovationS = (_Ck * _Pk * _Ck.transpose() + _Qgama);
    _InverseSolver.compute(_innovationS) ;
//    _Sinverse = _InverseSolver.solve( MatrixXs::Identity(_y_dim_p, _y_dim_p));
    _Sinverse = _InverseSolver.solve( MatrixXs::Identity(_innovationS.rows(), _innovationS.rows()) );

    _Kk = _Pk * _Ck.transpose() * _Sinverse ;
}

template<class Scalar>
void EKF<Scalar>::initializeState( const Eigen::Ref<VectorXs> &stateInitial_)
{
    if ( !isXinitialized)
    {
        if ( stateInitial_.rows() == _x_dim_n && stateInitial_.cols() == 1)
        {
            _Xk = stateInitial_;
            isXinitialized = true;
        }
        else
        {
            std::cout<< "expected: "<<_x_dim_n <<" * "<< 1 <<" provided: "
                     << stateInitial_.rows()<< " * "<< stateInitial_.cols() <<std::endl;
            throw std::runtime_error("EKF::initializeState(): mismatched size when initilize state!");
        }
    }
    else
    {
        throw std::runtime_error("EKF::initializeState(): reinitialize state!");
    }
};

template<class Scalar>
void EKF<Scalar>::setQ_alpha(const Eigen::Ref<MatrixXs> & Q_alpha_)
{
    if ( Q_alpha_.rows() == _x_dim_n && Q_alpha_.cols() == _x_dim_n)
    {
        _Qalpha = Q_alpha_;
        isQalphaSet = true;
    }
    else
    {
        std::cout<< "expected: "<<_x_dim_n <<" * "<<_x_dim_n<<" provided: "
                 << Q_alpha_.rows()<< " * "<< Q_alpha_.cols() <<std::endl;
        throw std::runtime_error("EKF::setQ_alpha(): mismatched size when set Q_alpha!");
    }
};

template<class Scalar>
void EKF<Scalar>::setQ_beta(const Eigen::Ref<MatrixXs>& Q_beta_)
{
    if ( Q_beta_.rows() == _u_dim_m && Q_beta_.cols() == _u_dim_m)
    {
        _Qbeta = Q_beta_;
        isQbetaSet = true;
    }
    else
    {
        std::cout<< "expected: "<<_u_dim_m <<" * "<<_u_dim_m<<" provided: "
                 << Q_beta_.rows()<< " * "<< Q_beta_.cols() <<std::endl;
        throw std::runtime_error("EKF::setQ_beta(): mismatched size when set Q_bete!");
    }
};

template<class Scalar>
void EKF<Scalar>::setQ_gamma(const Eigen::Ref<MatrixXs>& Q_gama_)
{
    if ( Q_gama_.rows() == _y_dim_p && Q_gama_.cols() == _y_dim_p)
    {
        _Qgama = Q_gama_;
        isQgammaSet = true;
    }
    else
    {
        std::cout<< "expected: "<<_y_dim_p <<" * "<<_y_dim_p<<" provided: "
                 << Q_gama_.rows()<< " * "<< Q_gama_.cols() <<std::endl;
        throw std::runtime_error("EKF::setQ_gamma(): mismatched size when set Q_gamma!");
    }
};

template<class Scalar>
void EKF<Scalar>::initializeP( const Eigen::Ref<MatrixXs>& initialP_)
{
    if ( initialP_.rows() == _x_dim_n && initialP_.cols() == _x_dim_n)
    {
        _Pk = initialP_;
        isPinitialized = true;
    }
    else
    {
        std::cout<< "expected: "<<_x_dim_n <<" * "<<_x_dim_n<<" provided: "
                 << initialP_.rows()<< " * "<< initialP_.cols() <<std::endl;
        throw std::runtime_error("EKF::initializeP(): mismatched size when initilize P!");
    }
}


#endif
