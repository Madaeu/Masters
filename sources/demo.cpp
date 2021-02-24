//
// Created by madaeu on 2/17/21.
//
#include "pinhole_camera.h"
#include "camera_pyramid.h"

#include <iostream>

int main()
{
    PinholeCamera<float> camera(1.0, 1.0, 5.0, 5.0, 1392, 512);
    CameraPyramid<float> cameraPyr(camera, 4);

    return 0;
}

