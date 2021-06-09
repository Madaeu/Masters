//
// Created by madaeu on 5/1/21.
//

#ifndef MASTERS_UNIFORM_SAMPLER_H
#define MASTERS_UNIFORM_SAMPLER_H

#include <random>
#include <cmath>
#include <chrono>

namespace msc
{
    struct Point
    {
        int x;
        int y;
    };

    class UniformSampler
    {
    public:
        UniformSampler(int imageWidth, int imageHeight);

        std::vector<Point> samplePoints(int numberOfPoints);

    private:
        std::mt19937 generator_;
        std::uniform_real_distribution<double> uniformDistribution_;

        int imageWidth_;
        int imageHeight_;
    };

    UniformSampler::UniformSampler(int imageWidth, int imageHeight)
    : uniformDistribution_(0.0, 1.0), imageWidth_(imageWidth), imageHeight_(imageHeight)
    {
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        generator_ = std::mt19937(seed);
    }

    std::vector<Point> UniformSampler::samplePoints(int numberOfPoints)
    {
        std::vector<Point> points;
        for (int i = 0; i < numberOfPoints; ++i)
        {
            Point point;
            point.x = imageWidth_ * uniformDistribution_(generator_);
            point.y = imageHeight_ * uniformDistribution_(generator_);
            points.push_back(point);
        }
        return points;
    }

} //namespace msc

#endif //MASTERS_UNIFORM_SAMPLER_H
