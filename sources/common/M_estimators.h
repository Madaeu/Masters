//
// Created by madaeu on 4/14/21.
//

#ifndef MASTERS_M_ESTIMATORS_H
#define MASTERS_M_ESTIMATORS_H

#include "Eigen/Dense"

namespace msc
{
    template<typename Scalar>
    EIGEN_DEVICE_FUNC
    Scalar cauchyWeight(Scalar x, Scalar delta){
        Scalar  a = delta/x;
        return abs(a)/ sqrt(2.)* sqrt(log(1+1/a/a));
    }

    template<typename Scalar>
    EIGEN_DEVICE_FUNC
    Scalar huberWeight(Scalar x, Scalar delta){
        const Scalar a = abs(x);
        return a <= delta ? Scalar(1): sqrt(delta * (2*a-delta)) / a;
    }
} //namespace msc

#endif //MASTERS_M_ESTIMATORS_H
