/*** Include ***/
/* for general */
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>

/* for OpenCV */
#include <opencv2/opencv.hpp>
#include <opencv2/tracking/tracker.hpp>

/*** Macro ***/
/* Settings */
typedef struct {
	int x0;
	int y0;
	int x1;
	int y1;
} RECT;


static RECT s_selectedArea;
static bool isAreaSelected = false;

void mouseCallback(int event, int x, int y, int flags, void *userdata)
{
	static bool isDrag = false;
	static RECT s_mouseDrag;
	switch (event) {
	case cv::EVENT_LBUTTONDOWN:
		isDrag = true;
		s_mouseDrag.x0 = x;
		s_mouseDrag.y0 = y;
		break;
	
	case cv::EVENT_LBUTTONUP:
		isDrag = false;
		isAreaSelected = true;
	case cv::EVENT_MOUSEMOVE:
		if (isDrag) {
			s_mouseDrag.x1 = x;
			s_mouseDrag.y1 = y;
			s_selectedArea = s_mouseDrag;
			if (s_selectedArea.x0 > s_selectedArea.x1) {
				int temp = s_selectedArea.x0;
				s_selectedArea.x0 = s_selectedArea.x1;
				s_selectedArea.x1 = temp;
			}
			if (s_selectedArea.y0 > s_selectedArea.y1) {
				int temp = s_selectedArea.y0;
				s_selectedArea.y0 = s_selectedArea.y1;
				s_selectedArea.y1 = temp;
			}
		}
		break;
	default:
		break;
	}
}
inline cv::Ptr<cv::Tracker> createTrackerByName(cv::String name)
{
	cv::Ptr<cv::Tracker> tracker;

	if (name == "KCF")
		tracker = cv::TrackerKCF::create();
	else if (name == "TLD")
		tracker = cv::TrackerTLD::create();
	else if (name == "BOOSTING")
		tracker = cv::TrackerBoosting::create();
	else if (name == "MEDIAN_FLOW")
		tracker = cv::TrackerMedianFlow::create();
	else if (name == "MIL")
		tracker = cv::TrackerMIL::create();
	else if (name == "GOTURN")
		tracker = cv::TrackerGOTURN::create();
	else if (name == "MOSSE")
		tracker = cv::TrackerMOSSE::create();
	else
		CV_Error(cv::Error::StsBadArg, "Invalid tracking algorithm name\n");

	return tracker;
}

int main()
{
	/*** Initialize ***/
	/* Initialize camera */
	int originalImageWidth = 1280;
	int originalImageHeight = 720;
	cv::VideoCapture cap;
	cap = cv::VideoCapture(0);
	cap.set(cv::CAP_PROP_FRAME_WIDTH, originalImageWidth);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, originalImageHeight);

	/* Initialize list for tracker */
	std::list<cv::Ptr<cv::Tracker>> trackerList;

	while (1) {
		/*** Read image ***/
		cv::Mat mat;
		cap.read(mat);

		cv::setMouseCallback("window", mouseCallback);
		
		cv::rectangle(mat, cv::Rect(s_selectedArea.x0, s_selectedArea.y0, s_selectedArea.x1 - s_selectedArea.x0, s_selectedArea.y1 - s_selectedArea.y0), cv::Scalar(0, 255, 0), 3);
		if (isAreaSelected) {
			/* Add a new tracker for the selected area */
			isAreaSelected = false;
			auto tracker = createTrackerByName("KCF");
			tracker->init(mat, cv::Rect(s_selectedArea.x0, s_selectedArea.y0, s_selectedArea.x1 - s_selectedArea.x0, s_selectedArea.y1 - s_selectedArea.y0));
			trackerList.push_back(tracker);
		}

		for (auto tracker : trackerList) {
			cv::Rect2d trackedRect;
			if (tracker->update(mat, trackedRect)) {
				cv::rectangle(mat, trackedRect, cv::Scalar(255, 0, 0), 2, 1);
			} else {
				printf("fail\n");
				trackerList.remove(tracker);
				tracker.release();
			}
		}
			
		cv::imshow("window", mat);
		if (cv::waitKey(1) == 'q') break;
	}
	for (auto tracker : trackerList) {
		tracker.release();
	}

	return 0;
}
