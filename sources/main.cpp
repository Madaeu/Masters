#include <iostream>

#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/ccalib.hpp>

static void drawOptFlowMap(const cv::Mat& flow, cv::Mat& cflowmap, int step,
                           double, const cv::Scalar& color)
{
    for(int y = 0; y < cflowmap.rows; y += step)
        for(int x = 0; x < cflowmap.cols; x += step)
        {
            const cv::Point2f& fxy = flow.at<cv::Point2f>(y, x);
            line(cflowmap, cv::Point(x,y), cv::Point(cvRound(x+fxy.x), cvRound(y+fxy.y)),
                 color);
            circle(cflowmap, cv::Point(x,y), 2, color, -1);
        }
}
int main(){
    cv::VideoCapture cap;
    cap.open("/home/madaeu/Masters/data/2011_09_26_drive_0002_sync/2011_09_26/2011_09_26_drive_0002_sync/image_03/data/%10d.png");
    if(!cap.isOpened()){
        std::cerr << "ERROR! Unable to open image sequence\n";
        return -1;
    }

    cv::Mat flow, cflow, frame, transform;
    cv::UMat gray, prevgray, uflow;

    cv::namedWindow("flow", 1);

    for(;;){
        cap >> frame;

        if(frame.empty()){
            break;
        }

        cv::cvtColor(frame, gray, cv::COLOR_RGB2GRAY);

        if(!prevgray.empty()){
            cv::calcOpticalFlowFarneback(prevgray, gray, uflow, 0.5, 3, 15, 3, 5, 1.2, 0);
            cv::cvtColor(prevgray, cflow, cv::COLOR_GRAY2BGR);
            uflow.copyTo(flow);
            drawOptFlowMap(flow, cflow, 16, 1.5, cv::Scalar(0, 255, 0));
            cv::imshow("flow", cflow);

        }
        cv::waitKey(100);
        std::swap(prevgray, gray);
    }
    return 0;
}

