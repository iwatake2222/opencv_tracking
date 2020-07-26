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
	int x;
	int y;
	int width;
	int height;
} RECT;

typedef struct {
	cv::Ptr<cv::Tracker> tracker;
	int numLost;
	RECT rectFirst;
} OBJECT_TRACKER;

enum {
	STATUS_NONE,
	STATUS_DRAG,
	STATUS_SELECTED,
} s_status;
static RECT s_selectedArea;

void mouseCallback(int event, int mouseX, int mouseY, int flags, void *userdata)
{
	
	static bool isDrag = false;
	static cv::Point s_dragStart;
	switch (event) {
	case cv::EVENT_LBUTTONDOWN:
		s_status = STATUS_DRAG;
		s_dragStart.x = mouseX;
		s_dragStart.y = mouseY;
		break;
	
	case cv::EVENT_LBUTTONUP:
		s_status = STATUS_SELECTED;
	case cv::EVENT_MOUSEMOVE:
		if (s_status == STATUS_DRAG) {
			s_selectedArea.x = std::min(mouseX, s_dragStart.x);
			s_selectedArea.y = std::min(mouseY, s_dragStart.y);
			s_selectedArea.width = std::abs(mouseX - s_dragStart.x);
			s_selectedArea.height = std::abs(mouseY - s_dragStart.y);
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
	std::vector<OBJECT_TRACKER> objectList;

	while (1) {
		/*** Read image ***/
		cv::Mat mat;
		cap.read(mat);

		cv::setMouseCallback("window", mouseCallback);
		
		cv::rectangle(mat, cv::Rect(s_selectedArea.x, s_selectedArea.y, s_selectedArea.width, s_selectedArea.height), cv::Scalar(0, 255, 0), 3);
		if (s_status == STATUS_SELECTED) {
			/* Add a new tracker for the selected area */
			s_status = STATUS_NONE;
			OBJECT_TRACKER object;
			object.tracker = createTrackerByName("KCF");
			object.tracker->init(mat, cv::Rect(s_selectedArea.x, s_selectedArea.y, s_selectedArea.width, s_selectedArea.height));
			object.numLost = 0;
			object.rectFirst = s_selectedArea;
			objectList.push_back(object);
		}
		for (auto it = objectList.begin(); it != objectList.end();) {
			auto tracker = it->tracker;
			cv::Rect2d trackedRect;
			if (tracker->update(mat, trackedRect)) {
				cv::rectangle(mat, trackedRect, cv::Scalar(255, 0, 0), 2, 1);
				it->numLost = 0;
				it++;
			} else {
				printf("lost\n");
				if (++(it->numLost) > 100) {
					printf("delete\n");
					it = objectList.erase(it);
					tracker.release();
				} else {
					it++;
				}
			}
		}
		
		cv::imshow("window", mat);
		if (cv::waitKey(1) == 'q') break;
	}
	for (auto obj : objectList) {
		obj.tracker.release();
	}

	return 0;
}
