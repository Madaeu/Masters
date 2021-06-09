//
// Created by madaeu on 5/5/21.
//

#ifndef MASTERS_SLAM_SYSTEM_OPTIONS_H
#define MASTERS_SLAM_SYSTEM_OPTIONS_H

#include "mapping/mapper.h"

#include <vector>
#include <string>

namespace msc
{
    struct SlamSystemOptions
    {
        using ConnectionMode = typename MapperOptions::ConnectionMode;
        enum KeyframeMode {AUTO=0, AUTO_COMBINED, NEVER};
        enum TrackerMode {CLOSEST = 0, LAST, FIRST};

        std::size_t gpu = 0;
        std::string networkPath;
        std::string vocabularyPath;

        /* Camera Tracker */
        std::vector<int> trackingIterations = {4, 6, 5, 10};
        TrackerMode trackingMode = CLOSEST;
        float trackingHuber = 0.3f;
        float trackingErrorThreshold = 0.3f;
        float trackingDistanceThreshold = 2;

        /* Keyframe Connections */
        ConnectionMode connectionMode = ConnectionMode::LAST;
        std::size_t maxBackConnections = 4;

        /* Keyframe Add Method */
        KeyframeMode keyframeMode = AUTO;
        float inlierThreshold = 0.5f;
        float distanceThreshold = 2.0f;
        float frameDistanceThreshold = 0.2f;
        float combinedThreshold = 2.0f;

        /* Loop Closure */
        bool loopClosure = true;
        float loopMaxDistance = 0.5f;
        int loopActiveWindows = 10;
        float loopSigma = 1.0f;
        float loopMinSimilarity = 0.35f;
        int loopMaxCandidates = 10;

        /* Mapping Parameters */
        bool interleaveMapping = false;
        float relinearizeSkip = 1;
        float relinearizeThreshold = 0.05f;
        float partialRelinearizeCheck = true;
        float posePrior = 0.3f;
        float codePrior = 1.0f;
        bool predictCode = true;

        /* Photometric Error */
        bool usePhotometric = true;
        std::vector<int> photometricIterations = {15, 15, 15, 30};
        float photometricHuber = 0.3f;
        int sfmStepThreads = 32;
        int sfmStepBlocks = 11;
        int sfmEvalThreads = 224;
        int sfmEvalBlocks = 66;
        bool normalizeImage = false;

        /* Reprojection Error */
        bool useReprojection = false;
        int reprojectionNFeatures = 500;
        float reprojectionScaleFactor = 1.2f;
        int reprojectionNLevels = 1;
        float reprojectionMaxDistance = 30.0f;
        float reprojectionHuber = 0.1f;
        int reprojectionIterations = 15;
        float reprojectionSigma = 1.0f;
        int reprojectionRansacIterations = 1000;
        float reprojectionRansacThreshold = 0.0001f;

        static KeyframeMode keyframeModeTranslator(const std::string& s);
        static std::string keyframeModeTranslator(KeyframeMode mode);
        static TrackerMode trackerModeTranslator(const std::string& s);
        static std::string trackerModeTranslator(TrackerMode mode);
        static ConnectionMode connectionModeTranslator(const std::string s);
        static std::string connectionModeTranslator(ConnectionMode mode);

    };

    SlamSystemOptions::KeyframeMode SlamSystemOptions::keyframeModeTranslator(const std::string &s)
    {
        if(s == "AUTO")
            return KeyframeMode::AUTO;
        else if(s == "NEVER")
            return KeyframeMode::NEVER;
        else if (s == "AUTO_COMBINED")
            return KeyframeMode::AUTO_COMBINED;
        else
            throw std::runtime_error("Unknown Keyframe Mode:" + s);
    }

    std::string SlamSystemOptions::keyframeModeTranslator(KeyframeMode mode)
    {
        std::string name;
        switch (mode)
        {
            case KeyframeMode::AUTO:
                name = "AUTO";
                break;
            case KeyframeMode::NEVER:
                name = "NEVER";
                break;
            case KeyframeMode::AUTO_COMBINED:
                name = "AUTO_COMBINED";
                break;
            default:
                name = "UNKNOWN";
                break;
        }
        return name;
    }

    SlamSystemOptions::TrackerMode SlamSystemOptions::trackerModeTranslator(const std::string &s)
    {
        if (s == "FIRST")
            return TrackerMode::FIRST;
        else if (s == "LAST")
            return TrackerMode::LAST;
        else if (s == "CLOSEST")
            return TrackerMode::CLOSEST;
        else
            throw std::runtime_error("Invalid Tracker mode: " + s);
    }

    std::string SlamSystemOptions::trackerModeTranslator(TrackerMode mode)
    {
        std::string name;
        switch (mode)
        {
            case TrackerMode::FIRST:
                name = "FIRST";
                break;
            case TrackerMode::LAST:
                name = "LAST";
                break;
            case TrackerMode::CLOSEST:
                name = "CLOSEST";
                break;
            default:
                name = "UNKNOWN";
                break;
        }
        return name;
    }

    SlamSystemOptions::ConnectionMode SlamSystemOptions::connectionModeTranslator(const std::string s)
    {
        if (s == "FIRST")
            return ConnectionMode::FIRST;
        else if (s == "LAST")
            return ConnectionMode::LAST;
        else if (s == "LASTN")
            return ConnectionMode::LASTN;
        else if (s == "FULL")
            return ConnectionMode::FULL;
        else
            throw std::runtime_error("Invalid Connection Mode: " + s);
    }

    std::string SlamSystemOptions::connectionModeTranslator(ConnectionMode mode)
    {
        std::string name;
        switch (mode)
        {
            case ConnectionMode::FIRST:
                name = "FIRST";
                break;
            case ConnectionMode::LAST:
                name = "LAST";
                break;
            case ConnectionMode::LASTN:
                name = "LASTN";
                break;
            case ConnectionMode::FULL:
                name = "FULL";
                break;
            default:
                name = "UNKNOWN";
                break;
        }
        return name;
    }

}


#endif //MASTERS_SLAM_SYSTEM_OPTIONS_H
