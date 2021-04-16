//
// Created by madaeu on 2/15/21.
//

#ifndef MASTERS_PINHOLE_CAMERA_H
#define MASTERS_PINHOLE_CAMERA_H

#include "Eigen/Core"

namespace msc {
    template<typename Scalar>
    class PinholeCamera {
    public:
        using PointT = Eigen::Matrix<Scalar, 3, 1>;
        using PixelT = Eigen::Matrix<Scalar, 2, 1>;
        using PointTRef = Eigen::Ref<const PointT>;
        using PixelTRef = Eigen::Ref<const PixelT>;

        /**
         *  Default constructor for PinholeCamera class.
         */
        EIGEN_DEVICE_FUNC PinholeCamera();

        /**
         * Constructor using camera intrinsics
         * @param fx
         * @param fy
         * @param u0
         * @param v0
         * @param width
         * @param height
         */
        EIGEN_DEVICE_FUNC PinholeCamera(Scalar fx, Scalar fy, Scalar u0, Scalar v0, Scalar width, Scalar height);

        /**
         * Destructor
         */
        EIGEN_DEVICE_FUNC ~PinholeCamera();

        /**
         * Calculate forward projection of 3D point onto image plane of camera
         * @param 3D point
         * @return
         */
        inline EIGEN_DEVICE_FUNC PixelT forwardProjection(const PointTRef &point) const;

        /**
         * Calculate the Jacobian of forward projection w.r.t. 3D point
         * @param 3D point
         * @return
         */
        inline EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 2, 3> forwardProjectionJacobian(const PointTRef &point) const;

        /**
         * Calculate backward projection of 2D point to a given depth in Euclidean space.
         * @param pixel
         * @param depth
         * @return
         */
        inline EIGEN_DEVICE_FUNC PointT backwardProjection(const PixelTRef &pixel, Scalar depth) const;

        /**
         * Calculate the Jacobian of backward projection w.r.t 2D point
         * @param pixel
         * @param depth
         * @return
         */
        inline EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 3, 2>
        backwardProjectionPointJac(const PixelTRef &pixel, Scalar depth) const;

        /**
         * Calculate the Jacobian of backward projection w.r.t depth
         * @param pixel
         * @param depth
         * @return
         */
        inline EIGEN_DEVICE_FUNC Eigen::Matrix<Scalar, 3, 1>
        backwardProjectionDepthJac(const PixelTRef &pixel, Scalar depth) const;

        /**
         * Validate that pixel is within camera's image plane
         * @param x
         * @param y
         * @param border
         * @return
         */
        inline EIGEN_DEVICE_FUNC bool isPixelValid(Scalar x, Scalar y, std::size_t border = 0) const;

        /**
         * Validate that pixel is within camera's image plane
         * @param pixel
         * @param border
         * @return
         */
        inline EIGEN_DEVICE_FUNC bool isPixelValid(const PixelTRef &pixel, std::size_t border = 0) const;

        /**
         * Resize camera intrinsics to image dimensions
         * @tparam T
         * @param new_width
         * @param new_height
         */
        template<typename T>
        inline void resizeViewport(const T &new_width, const T &new_height);

        template<typename T>
        inline Eigen::Matrix<T, 3, 3> Matrix() const;

        template<typename T>
        inline Eigen::Matrix<T, 3, 3> InvMatrix() const;

        template<typename T>
        inline PinholeCamera<T> Cast() const;

        inline const Scalar &fx() const { return fx_; }

        inline const Scalar &fy() const { return fy_; }

        inline const Scalar &u0() const { return u0_; }

        inline const Scalar &v0() const { return v0_; }

        inline const Scalar &width() const { return width_; }

        inline const Scalar &height() const { return height_; }


    private:
        Scalar fx_{};
        Scalar fy_{};
        Scalar u0_{};
        Scalar v0_{};
        Scalar width_{};
        Scalar height_{};
    };

    template<typename Scalar>
    PinholeCamera<Scalar>::PinholeCamera():
            PinholeCamera(0, 0, 0, 0, 0, 0) {}

    template<typename Scalar>
    PinholeCamera<Scalar>::PinholeCamera(Scalar fx, Scalar fy, Scalar u0, Scalar v0, Scalar width, Scalar height)
            : fx_(fx), fy_(fy), u0_(u0), v0_(v0), width_(width), height_(height) {

    }

    template<typename Scalar>
    PinholeCamera<Scalar>::~PinholeCamera() {

    }

    template<typename Scalar>
    typename PinholeCamera<Scalar>::PixelT
    PinholeCamera<Scalar>::forwardProjection(const PinholeCamera::PointTRef &point) const {
        return PixelT(fx_ * point[0] / point[2] + u0_,
                      fy_ * point[1] / point[2] + v0_);
    }

    template<typename Scalar>
    Eigen::Matrix<Scalar, 2, 3>
    PinholeCamera<Scalar>::forwardProjectionJacobian(const PinholeCamera::PointTRef &point) const {
        Eigen::Matrix<Scalar, 2, 3> jacobian;
        jacobian << fx_ / point[2],              0, -(fx_ * point[0]) / point[2] / point[2],
                                 0, fy_ / point[2], -(fy_ * point[1]) / point[2] / point[2];
        return jacobian;
    }

    template<typename Scalar>
    typename PinholeCamera<Scalar>::PointT
    PinholeCamera<Scalar>::backwardProjection(const PinholeCamera::PixelTRef &pixel, Scalar depth) const {
        PointT point((pixel[0] - u0_) / fx_,
                     (pixel[1] - v0_) / fy_, 1);
        return depth * point;
    }

    template<typename Scalar>
    Eigen::Matrix<Scalar, 3, 2>
    PinholeCamera<Scalar>::backwardProjectionPointJac(const PinholeCamera::PixelTRef &pixel, Scalar depth) const {
        Eigen::Matrix<Scalar, 3, 2> jacobian;
        jacobian << depth / fx_,           0,
                              0, depth / fy_,
                              0,           0;
        return jacobian;
    }

    template<typename Scalar>
    Eigen::Matrix<Scalar, 3, 1>
    PinholeCamera<Scalar>::backwardProjectionDepthJac(const PinholeCamera::PixelTRef &pixel, Scalar depth) const {
        Eigen::Matrix<Scalar, 3, 1> jacobian;
        jacobian << (pixel[0] - u0_) / fx_,
                    (pixel[1] - v0_) / fy_,
                                         1;
        return jacobian;
    }

    template<typename Scalar>
    bool PinholeCamera<Scalar>::isPixelValid(Scalar x, Scalar y, std::size_t border) const {
        return (x >= border) && (x < width_ - border) && (y >= border) && (y < height_ - border);
    }

    template<typename Scalar>
    bool PinholeCamera<Scalar>::isPixelValid(const PinholeCamera::PixelTRef &pixel, std::size_t border) const {
        return isPixelValid(pixel[0], pixel[1], border);
    }

    template<typename Scalar>
    template<typename T>
    void PinholeCamera<Scalar>::resizeViewport(const T &new_width, const T &new_height) {
        const Scalar x_ratio{new_width / width_};
        const Scalar y_ratio{new_height / height_};
        fx_ *= x_ratio;
        fy_ *= y_ratio;
        u0_ *= x_ratio;
        v0_ *= y_ratio;
        width_ = new_width;
        height_ = new_height;
    }

    template<typename Scalar>
    template<typename T>
    Eigen::Matrix<T, 3, 3> PinholeCamera<Scalar>::Matrix() const {
        Eigen::Matrix<T, 3, 3> intrinsics = Eigen::Matrix<T, 3, 3>::Identity();
        intrinsics(0, 0) = fx();
        intrinsics(0, 2) = u0();
        intrinsics(1, 1) = fy();
        intrinsics(1, 2) = v0();
        return intrinsics;
    }

    template<typename Scalar>
    template<typename T>
    Eigen::Matrix<T, 3, 3> PinholeCamera<Scalar>::InvMatrix() const {
        Eigen::Matrix<T, 3, 3> invIntrinsics = Eigen::Matrix<T, 3, 3>::Identity();
        invIntrinsics(0,0) = 1.0/fx();
        invIntrinsics(0,2) = -u0()/fx();
        invIntrinsics(1,1) = 1.0/fy();
        invIntrinsics(1,2) = -v0()/fy();
        return invIntrinsics;
    }

    template<typename Scalar>
    template<typename T>
    PinholeCamera<T> PinholeCamera<Scalar>::Cast() const {
        return PinholeCamera<T>(T(fx_), T(fy_), T(u0_), T(v0_), width_, height_);
    }
} //namespace msc

#endif //MASTERS_PINHOLE_CAMERA_H
