#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"


int main() {
    //Setup images and vectors for keypoints and descriptors.
    cv::Mat img1, img2;
    img1 = cv::imread("/home/nicklas/Desktop/img1.png", 0);
    img2 = cv::imread("/home/nicklas/Desktop/img2.png", 0);
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
    double max_dist, min_dist;
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

    std::cout << "total matches: " << matches.size() << std::endl;
    std::cout << "good matches: " << good_matches.size() << std::endl;
    std::cout << "sel_img1 size: " << sel_img1.size() << std::endl;
    std::cout << "sel_img2 size: " << sel_img1.size() << std::endl;

    //Draw matches
    cv::Mat matchImg;
    cv::drawMatches(img1, keypointsimg1, img2, keypointsimg2, good_matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                    cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches", matchImg);
    cv::waitKey(0);
    return 0;
}