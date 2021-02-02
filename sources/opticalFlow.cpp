#include <iostream>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/calib3d.hpp>

using namespace cv;
int main(){
    Point2f point;
    bool addRemovePt = false;

    double intrinsic[3][3] = {{9.037596e+02, 0.000000e+00, 6.957519e+02},{0.000000e+00, 9.019653e+02, 2.242509e+02},{0.000000e+00, 0.000000e+00, 1.000000e+00}};
    Mat camMatrix(3,3, CV_64F, intrinsic);

    VideoCapture cap;
    cap.open("/home/madaeu/Masters/data/2011_09_26_drive_0002_sync/2011_09_26/2011_09_26_drive_0002_sync/image_03/data/%10d.png");
    if(!cap.isOpened()){
        std::cerr << "ERROR! Unable to open image sequence\n";
        return -1;
    }

    TermCriteria termCrit(TermCriteria::COUNT|TermCriteria::EPS, 20, 0.03);
    Size subPixWinSize(10, 10), winSize(31,31);

    const int MAX_COUNT = 500;
    bool needToInit = true;

    namedWindow("LKOptflow", 1);

    Mat gray, prevGray, image, frame;
    std::vector<Point2f> points[2];

    for(;;){
        cap >> frame;
        if(frame.empty())
            break;
        frame.copyTo(image);
        cvtColor(image, gray, COLOR_RGB2GRAY);

        if( needToInit){
            goodFeaturesToTrack(gray, points[1], MAX_COUNT, 0.01, 10, Mat(), 3, 0, 0.04);
            cornerSubPix(gray, points[1], subPixWinSize, Size(-1,-1), termCrit);
            addRemovePt = false;
        }
        else if( !points[0].empty()){
            std::vector<uchar> status;
            std::vector<float> err;
            if(prevGray.empty())
                gray.copyTo(prevGray);
            calcOpticalFlowPyrLK(prevGray, gray, points[0], points[1], status, err, winSize, 3, termCrit, 0, 0.001);
            size_t i, k;
            for (i = k = 0; i < points[1].size(); i++ ){
                if (!status[i])
                    continue;
                points[1][k++] = points[1][i];
                circle( image, points[1][i], 3, Scalar(0,255,0), -1, 8);
            }
            Mat E, R, t, mask;
            E = findEssentialMat(points[0], points[1], camMatrix, RANSAC, 0.999, 1.0, mask);
            recoverPose(E, points[0], points[1], camMatrix, R, t, mask);
            std::cout << R << "\n " << t << std::endl;

            points[1].resize(k);

        }
        needToInit = false;

        imshow("LKOptflow", image);
        char c = (char)waitKey();
        if( c == 27 )
            break;
        switch( c )
        {
            case 'r':
                needToInit = true;
                break;
            case 'c':
                points[0].clear();
                points[1].clear();
                break;
        }
        std::swap(points[1], points[0]);
        cv::swap(prevGray, gray);
        }
}
