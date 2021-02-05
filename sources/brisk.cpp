#include <iostream>
#include <fstream>
#include <vector>
#include <opencv2/core.hpp>
#include "opencv2/opencv.hpp"
#include "opencv2/features2d/features2d.hpp"
#include <ctime>

//Camera intrinsics parameters
double fx = 9.842439e+02;
double cx = 6.900000e+02;
double fy = 9.808141e+02;
double cy = 2.331966e+02;

cv::Mat K = (cv::Mat1d(3,3) << fx, 0, cx, 0, fy, cy, 0, 0, 1);
cv::Mat dist = (cv::Mat1d(5,1) << -3.728755e-01, 2.037299e-01, 2.219027e-03, 1.383707e-03, -7.233722e-02);
struct pose{
    cv::Mat R, t;
};

//Setup BRISK parameters
int Threshl = 80;
int Octaves = 4;
float PatternScales = 1.0f;

cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create(cv::DescriptorMatcher::BRUTEFORCE_HAMMING);

//Setup vectors for keypoints and mats for descriptors
std::vector<cv::KeyPoint> keypointsimg1, keypointsimg2;
cv::Mat descriptorsimg1, descriptorsimg2;

//Match filtering setup
double max_dist = 0;
double min_dist = 1000;

//Pose holders and mask
cv::Mat t, R, Mask;

pose getPoseDiff(cv::Mat img1, cv::Mat img2, bool drawMatches){

    std::vector<cv::DMatch> matches, good_matches;
    //Detect features
    cv::Ptr<cv::BRISK> brisk = cv::BRISK::create(Threshl, Octaves, PatternScales);
    brisk->detect(img1, keypointsimg1);
    brisk->detect(img2, keypointsimg2);
    brisk->compute(img1, keypointsimg1, descriptorsimg1);
    brisk->compute(img2, keypointsimg2, descriptorsimg2);

    //Match
    matcher->match(descriptorsimg1, descriptorsimg2, matches);

    //Match filtering
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

    if(drawMatches == true){
        //Draw matches
        cv::Mat matchImg;
        cv::drawMatches(img1, keypointsimg1, img2, keypointsimg2, good_matches,
                        matchImg, cv::Scalar::all(-1), cv::Scalar::all(-1), std::vector<char>(),
                        cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::imshow("Matches", matchImg);
        cv::waitKey(0);}

    std::vector<cv::Point2f> sel_img1, sel_img2;

    //Sort keypoints in images so only relevant keypoints are remaining
    for(int i=0;i<good_matches.size();i++){
        sel_img1.push_back(keypointsimg1[good_matches[i].queryIdx].pt);
        sel_img2.push_back(keypointsimg2[good_matches[i].trainIdx].pt);
    }

    cv::Mat E = cv::findEssentialMat(sel_img1, sel_img2, K, cv::RANSAC, 0.99, 0.1, Mask);
    cv::recoverPose(E, sel_img1, sel_img2, K, R, t, Mask);
    return {R.inv(), (-1*t)};
}

int main() {

    clock_t time_req;
    std::vector<cv::Mat> Transforms;
    std::vector<cv::Mat> Rotations, Translations;
    //Setup images
    cv::Mat img1, img2, img1d, img2d;
    cv::String directory = "/home/nicklas/Desktop/2011_09_26_drive_0002_extract/2011_09_26/2011_09_26_drive_0002_extract/image_00/data/*.png";
    std::vector<cv::String> filenames;
    cv::glob(directory, filenames, false);

    time_req=clock();
    for(int i = 0; i<(filenames.size()-1); i++) {
        img1 = cv::imread(filenames[i], 0);
        img2 = cv::imread(filenames[i + 1], 0);
        cv::undistort(img1, img1d, K, dist, cv::noArray());
        cv::undistort(img2, img2d, K, dist, cv::noArray());
        auto[R, t] = getPoseDiff(img1d, img2d, false);
        Rotations.push_back(R);
        Translations.push_back(t);

        //Create Homogeneous transformation from R and t
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
    time_req = clock() - time_req;
    std::cout << "Finished pose path generation in: " << (float)time_req/CLOCKS_PER_SEC << " seconds" << std::endl;
    std::cout << "FPS: " << Rotations.size()/((float)time_req/CLOCKS_PER_SEC) << std::endl;
    //Draw trajectory
    cv::Mat Trajectory = cv::Mat::zeros(600, 600, CV_8UC3);
    cv::Mat T_f = Transforms[0];
    cv::Mat t_f;
    double fontScale = 1;
    char text[100];
    cv::Point textOrg(10, 50);
    int fontFace = cv::FONT_HERSHEY_PLAIN;
    int thickness = 1;

    for (int i = 1; i < Transforms.size(); ++i) {
        T_f = T_f * Transforms[i];

        int x = int(T_f.at<double>(0, 3)) + 300;
        int y = int(T_f.at<double>(2, 3)) + 300;
        circle(Trajectory, cv::Point(x, y) ,1, CV_RGB(255,0,0), 2);

        rectangle(Trajectory, cv::Point(10, 30), cv::Point(550, 50), CV_RGB(0,0,0), CV_FILLED);
        sprintf(text, "Coordinates: x = %02fm y = %02fm z = %02fm", T_f.at<double>(0, 3), T_f.at<double>(1, 3), T_f.at<double>(2, 3));
        putText(Trajectory, text, textOrg, fontFace, fontScale, cv::Scalar::all(255), thickness, 8);
        cv::imshow("Trajectory: ", Trajectory);
        cv::waitKey(100);
    }
    //Save resulting trajectory
    //cv::imwrite("/home/nicklas/Desktop/MasterResults/VOtrajectories/2011_09_26_drive_0002_trajectory.PNG", Trajectory);

    //Write transforms to file
    /* std::ofstream FileHandler;
     FileHandler.open("File_1.txt");
     for(int i = 0; i<Rotations.size(); i++){
         //FileHandler << Rotations[i] << std::endl;
         //FileHandler << Translations[i] << std::endl;
           FileHandler << Transforms[i] << "," << std::endl; // Write T to file
     }
     FileHandler.close();
 */
    //Sanity checks
    std::cout << "Rotations: " << Rotations.size() << ", Translations: " << Translations.size() << std::endl;
    std::cout << "trans: " << Transforms.size() << std::endl;

    return 0;
}

