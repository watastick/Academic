
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/video/tracking.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#define OPENCV_VERSION(a,b,c) (((a) << 16) + ((b) << 8) + (c))
#define OPENCV_VERSION_CODE OPENCV_VERSION(CV_MAJOR_VERSION, CV_MINOR_VERSION, CV_SUBMINOR_VERSION)

int
main(int argc, char *argv[])
{
	// image1
	cv::Mat img1 = cv::imread("INPUT SCENE1 IMAGE.png", 1);
	if (img1.empty()) return -1;
	// image2
	cv::Mat img2 = cv::imread("INPUT SCENE2 IMAGE.png", 1);
	if (img2.empty()) return -1;

	cv::Mat prev, next;
	cv::cvtColor(img1, prev, CV_BGR2GRAY);
	cv::cvtColor(img2, next, CV_BGR2GRAY);

	std::vector<cv::Point2f> prev_pts;
	std::vector<cv::Point2f> next_pts;


	cv::Size flowSize(30, 30);
	cv::Point2f center = cv::Point(prev.cols / 2., prev.rows / 2.);
	for (int i = 0; i<flowSize.width; ++i) {
		for (int j = 0; j<flowSize.width; ++j) {
			cv::Point2f p(i*float(prev.cols) / (flowSize.width - 1),
				j*float(prev.rows) / (flowSize.height - 1));
			prev_pts.push_back((p - center)*0.9f + center);
		}
	}

	// Lucas-Kanade
	// parameters=default
#if OPENCV_VERSION_CODE > OPENCV_VERSION(2,3,0)
	cv::Mat status, error;
#else
	std::vector<uchar> status;
	std::vector<float> error;
#endif
	cv::calcOpticalFlowPyrLK(prev, next, prev_pts, next_pts, status, error);

	
	std::vector<cv::Point2f>::const_iterator p = prev_pts.begin();
	std::vector<cv::Point2f>::const_iterator n = next_pts.begin();
	for (; n != next_pts.end(); ++n, ++p) {
		cv::line(img1, *p, *n, cv::Scalar(150, 0, 0), 2);
	}

	cv::namedWindow("optical flow", CV_WINDOW_AUTOSIZE | CV_WINDOW_FREERATIO);
	cv::imshow("optical flow", img1);
	cv::waitKey(0);
}


