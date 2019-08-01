#ifndef __VS_with_sensor_fusion_hpp__
#define __VS_with_sensor_fusion_hpp__


#include "VS_with_robot_joints.hpp"

template<class Scalar>
class VS_with_sensor_fusion : public VS_with_robot_joints<Scalar>
{
public:
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, 1> VectorXs;
    typedef Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic> MatrixXs;

    VS_with_sensor_fusion(
            int nPoints,
            Scalar dt,
            pinocchio::Model model_
            );

    void setSensorMatrix(const Eigen::Ref<VectorXs>& Yk_);
    void getStateUpdated (const Eigen::Ref<VectorXs>& Uk_) override;
    void setQ_gamma( const Eigen::Ref<MatrixXs>& Q_gama_) override;

private:
    MatrixXs _Qgama_fixed;

};

#include "ekf_vvs/VS_with_sensor_fusion.hxx"

#endif