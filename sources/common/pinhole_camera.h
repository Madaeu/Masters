//
// Created by madaeu on 2/15/21.
//

#ifndef MASTERS_PINHOLE_CAMERA_H
#define MASTERS_PINHOLE_CAMERA_H


template <typename Scalar> class PinholeCamera {
public:
    PinholeCamera() = default;
    PinholeCamera(Scalar fx, Scalar fy, Scalar u0, Scalar v0, Scalar width, Scalar height);
    virtual ~PinholeCamera();

    template <typename T>
    inline void resizeViewport(const T& new_width, const T& new_height);

    inline const Scalar& fx() const {return fx_;}
    inline const Scalar& fy() const {return fy_;}
    inline const Scalar& u0() const {return u0_;}
    inline const Scalar& v0() const {return v0_;}
    inline const Scalar& width() const {return width_;}
    inline const Scalar& height() const {return height_;}


private:
    Scalar fx_{};
    Scalar fy_{};
    Scalar u0_{};
    Scalar v0_{};
    Scalar width_{};
    Scalar height_{};
};

template<typename Scalar>
PinholeCamera<Scalar>::PinholeCamera(Scalar fx, Scalar fy, Scalar u0, Scalar v0, Scalar width, Scalar height)
: fx_(fx), fy_(fy), u0_(u0), v0_(v0), width_(width), height_(height){

}

template<typename Scalar>
PinholeCamera<Scalar>::~PinholeCamera() {

}

template<typename Scalar>
template<typename T>
void PinholeCamera<Scalar>::resizeViewport(const T &new_width, const T &new_height) {
    const Scalar x_ratio{new_width/width_};
    const Scalar y_ratio{new_height/height_};
    fx_ *= x_ratio;
    fy_ *= y_ratio;
    u0_ *= x_ratio;
    v0_ *= y_ratio;
    width_ = new_width;
    height_ = new_height;
}


#endif //MASTERS_PINHOLE_CAMERA_H
