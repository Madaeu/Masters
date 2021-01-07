#include <iostream>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"


int main() {
    //Vid stream
    /*std::vector<cv::KeyPoint> keypointsVid;
    cv::Mat vidFrame, keys;
    cv::VideoCapture vid("/home/nicklas/Downloads/sample-avi-file.avi");*/

    //Setup images and vectors for keypoints and descriptors.
    cv::Mat img1, img2;
    img1 = cv::imread("/home/nicklas/Pictures/and.png", 0);
    img2 = cv::imread("/home/nicklas/Pictures/and.png", 0);
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
    std::vector<cv::DMatch> matches;

    //Match
    matcher->match(descriptorsimg1, descriptorsimg2, matches);

    //INDSÆT MATCH FILTER
    /*float filter_thresh = 0.8f;
    std::vector<cv::DMatch> filtered_matches;
    for(int i=0; i<matches.size(); i++){
        if (matches[i].distance < filter_thresh * matches[i].distance)
        {
            filtered_matches.push_back(matches[i]);
        }
    }*/

    //Draw and show matches
    cv::Mat matchImg;
    cv::drawMatches(img1, keypointsimg1, img2, keypointsimg2, matches,
                    matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("Matches", matchImg);

    //Detect on vid stream
    /*while(true) {
        bool succes = vid.read(vidFrame);
        if(!succes)
        {
            std::cout << "No more vid :) ";
            break;
        }
        brisk->detect(vidFrame, keypointsVid);
        cv::drawKeypoints(vidFrame, keypointsVid, keys);
        cv::imshow("vidkeys", keys);
        if(cv::waitKey(30) == 27)
        {
            std::cout << "ESC key pressed: Exiting video";
            break;
        }
    }*/
    cv::waitKey(0);
    return 0;
}