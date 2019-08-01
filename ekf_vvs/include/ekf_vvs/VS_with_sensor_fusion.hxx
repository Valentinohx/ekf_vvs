

#ifndef __VS_with_sensor_fusion_hxx__
#define __VS_with_sensor_fusion_hxx__


template<class Scalar>
VS_with_sensor_fusion<Scalar>::VS_with_sensor_fusion(
        int nPoints,
        Scalar dt,
        pinocchio::Model model_
) : VS_with_robot_joints<Scalar>(nPoints, dt, model_){

    _Qgama_fixed = MatrixXs::Zero(this->_y_dim_p, this->_y_dim_p);
}

template<class Scalar>
void VS_with_sensor_fusion<Scalar>::setQ_gamma(const Eigen::Ref<MatrixXs>& Q_gama_)
{
    if ( Q_gama_.rows() == this->_y_dim_p && Q_gama_.cols() == this->_y_dim_p)
    {
        this->_Qgama = Q_gama_;
        _Qgama_fixed = this->_Qgama;
        this->isQgammaSet = true;
    }
    else
    {
        std::cout<< "expected: "<<this->_y_dim_p <<" * "<<this->_y_dim_p<<" provided: "
                 << Q_gama_.rows()<< " * "<< Q_gama_.cols() <<std::endl;
        throw std::runtime_error("VS_with_sensor_fusion::setQ_gamma(): mismatched size when set Q_gamma!");
    }
    std::cout<<"VS_with_sensor_fusion::setQ_gamma called" << std::endl;
};


template<class Scalar>
void VS_with_sensor_fusion<Scalar>::setSensorMatrix(const Eigen::Ref<VectorXs>& Yk_){

    int yn = Yk_.rows();
    this->_Qgama = MatrixXs::Zero(yn, yn);

    this->_Kk = MatrixXs::Zero( this->_x_dim_n, yn);
    this->_Yk = MatrixXs::Zero(yn, 1);
    this->_gXk = MatrixXs::Zero(yn, 1);
    this->_Ck = MatrixXs::Zero(yn,  this->_x_dim_n);
    this->_innovationS = MatrixXs::Zero(yn, yn);

    //set measurement equation
    this->_gXk =  this->_Xk.block(0, 0, yn, 1);

    //set Ck
    if( yn == 2*this->_nPoints )  //only feature is available
    {
        (this->_Ck.block(0, 0, yn, yn)).setIdentity();
        this->_Qgama = _Qgama_fixed.topLeftCorner(yn, yn);
    }
    else if( yn == this->_nJoints ) //only joints encoder is available
    {
        (this->_Ck.block(0, 2*this->_nPoints, yn, yn)).setIdentity();
        this->_Qgama = _Qgama_fixed.bottomRightCorner(yn, yn);
    }

    else if( yn == (2*this->_nPoints +  this->_nJoints) )  //both sensor are available
    {
        (this->_Ck.block(0, 0, 2*this->_nPoints, 2*this->_nPoints)).setIdentity();
        (this->_Ck.block(2*this->_nPoints, 2*this->_nPoints, this->_nJoints, this->_nJoints)).setIdentity();
        this->_Qgama = _Qgama_fixed;
    }
    else
    {
        throw std::runtime_error("VS_with_sensor_fusion::setSensorMatrix(): invalid measurement size!");
    }
}

template<class Scalar>
void VS_with_sensor_fusion<Scalar>::getStateUpdated(const Eigen::Ref<VectorXs>& Uk_)
{
    this->_Xk.block(0, 0, 2*this->_nPoints, 1) +=   this->_interaction_Lxy *  this->_camScrew * this->_dt;
    this->_Xk.block(2* this->_nPoints, 0,  this->_nJoints, 1) +=  Uk_ * this->_dt;
    this->_Xk.block(2* this->_nPoints+ this->_nJoints, 0,  this->_nPoints, 1) +=  this->_interaction_LZ *  this->_camScrew * this->_dt;
}


#endif
