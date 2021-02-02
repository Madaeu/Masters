#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"

double fx = 9.842439e+02;
double cx = 6.900000e+02;
double fy = 9.808141e+02;
double cy = 2.331966e+02;
cv::Mat K = (cv::Mat1d(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
struct pose{
    cv::Mat R, t;
};

pose getPoseDiff(cv::Mat img1, cv::Mat img2){


    //Setup vectors for keypoints and mats for descriptors
    std::vector<cv::KeyPoint> keypointsimg1, keypointsimg2;
    cv::Mat descriptorsimg1, descriptorsimg2;

    //Setup BRISK parameters
    int Threshl = 80;
    int Octaves = 4;
    float PatternScales = 1.0f;

    //Detect features
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(Threshl, Octaves, PatternScales);
    brisk->detect(img1, keypointsimg1);
    brisk->detect(img2, keypointsimg2);
    brisk->compute(img1, keypointsimg1, descriptorsimg1);
    brisk->compute(img2, keypointsimg2, descriptorsimg2);

    //Setup matcher
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);
    std::vector<cv::DMatch> matches, good_matches;
    std::vector<cv::Point2f> sel_img1, sel_img2;

    //Match
    matcher->match(descriptorsimg1, descriptorsimg2, matches);

    //Match filtering
    double max_dist = 0;
    double min_dist = 1000;
    for (int i =0; i < matches.size();i++)
    {
        if(max_dist<matches[i].distance) max_dist = matches[i].distance;
        if(min_dist>matches[i].distance) min_dist = matches[i].distance;
    }
    min_dist = min_dist + (max_dist- min_dist)* 0.3;
    for (int i=0;i<matches.size();i++)
    {
        if( matches[i].distance< min_dist)
            good_matches.push_back(matches[i]);
    }

    //Sort keypoints in images so only relevant keypoints are remaining
    for(int i=0;i<good_matches.size();i++){
        sel_img1.push_back(keypointsimg1[good_matches[i].queryIdx].pt);
        sel_img2.push_back(keypointsimg2[good_matches[i].trainIdx].pt);
    }
/*
    std::cout << "total matches: " << matches.size() << std::endl;
    std::cout << "good matches: " << good_matches.size() << std::endl;
    std::cout << "sel_img1 size: " << sel_img1.size() << std::endl;
    std::cout << "sel_img2 size: " << sel_img1.size() << std::endl;
*/
    cv::Mat t, R, Mask;
    cv::Mat E = cv::findEssentialMat(sel_img1, sel_img2, K, cv::RANSAC, 0.99, 0.1, Mask);
    cv::recoverPose(E, sel_img1, sel_img2, K, R, t, Mask);
    return {R, t};
/*
    //Draw matches
    cv::Mat matchImg;
    cv::drawMatches(img1, keypointsimg1, img2, keypointsimg2, good_matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches", matchImg);
    cv::waitKey(0);*/
}

int main() {

    std::vector<cv::Mat> Transforms;
    std::vector<cv::Mat> Rotations, Translations;
    //Setup images
    cv::Mat img1, img2;
    cv::String directory = "/home/nicklas/Desktop/2011_09_26_drive_0002_extract/2011_09_26/2011_09_26_drive_0002_extract/image_00/data/*.png";
    std::vector<cv::String> filenames;
    cv::glob(directory, filenames, false);
    for(int i = 0; i<(filenames.size()-1); i++) {
        img1 = cv::imread(filenames[i], 0);
        img2 = cv::imread(filenames[i + 1], 0);
        auto[R, t] = getPoseDiff(img1, img2);
        Rotations.push_back(R);
        Translations.push_back(t);

        cv::Mat T = (cv::Mat1d(4,4));
        T.at<double>(0, 0) = R.at<double>(0, 0);
        T.at<double>(0, 1) = R.at<double>(0, 1);
        T.at<double>(0, 2) = R.at<double>(0, 2);
        T.at<double>(1, 0) = R.at<double>(1, 0);
        T.at<double>(1, 1) = R.at<double>(1, 1);
        T.at<double>(1, 2) = R.at<double>(1, 2);
        T.at<double>(2, 0) = R.at<double>(2, 0);
        T.at<double>(2, 1) = R.at<double>(2, 1);
        T.at<double>(2, 2) = R.at<double>(2, 2);

        T.at<double>(0, 3) = t.at<double>(0, 0);
        T.at<double>(1, 3) = t.at<double>(0, 1);
        T.at<double>(2, 3) = t.at<double>(0, 2);

        T.at<double>(3, 0) = 0;
        T.at<double>(3, 1) =0;
        T.at<double>(3, 2) =0;
        T.at<double>(3, 3) =1;

        Transforms.push_back(T);
    }
    std::ofstream FileHandler;
    FileHandler.open("File_1.txt");
    for(int i = 0; i<Rotations.size(); i++){
       // FileHandler << Rotations[i] << std::endl;
       // FileHandler << Translations[i] << std::endl;
        FileHandler << Transforms[i] << std::endl;
    }
    FileHandler.close();
    std::cout << "Rotations: " << Rotations.size() << ", Translations: " << Translations.size() << std::endl;
    std::cout << "trans: " << Transforms.size() << std::endl;

    return 0;
}