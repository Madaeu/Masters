//
// Created by madaeu on 2/24/21.
//

#ifndef MASTERS_DATASET_INTERFACE_H
#define MASTERS_DATASET_INTERFACE_H

#include "camera_interface.h"

#include "opencv2/opencv.hpp"

struct DatasetFrame{
    double timestamp;
    cv::Mat img;
    cv::Mat dpt;
};

class DatasetInterface : public CameraInterface
{
public:
    DatasetInterface();
    virtual ~DatasetInterface();

    virtual std::vector<DatasetFrame> getAll() = 0;
    virtual bool hasPoses() = 0;
    virtual bool hasMore() = 0;
};
#endif //MASTERS_DATASET_INTERFACE_H
