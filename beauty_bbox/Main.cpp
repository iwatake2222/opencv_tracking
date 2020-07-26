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

void drawRing(cv::Mat &mat, RECT rect, cv::Scalar color, int animCount)
{
	/* Reference: https://github.com/Kazuhito00/object-detection-bbox-art */
	animCount *= int(135 / 30.0);
	cv::Point center(rect.x + rect.width / 2, rect.y + rect.height / 2);
	cv::Point radius((rect.width + rect.height) / 4, (rect.width + rect.height) / 4);
	int  ring_thickness = std::max(int(radius.x / 20), 1);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 80 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 150 + animCount, 0, 30, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 200 + animCount, 0, 10, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 230 + animCount, 0, 10, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 260 + animCount, 0, 60, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 337 + animCount, 0, 5, color, ring_thickness);

	radius *= 0.9;
	ring_thickness = std::max(int(radius.x / 12), 1);
	cv::ellipse(mat, cv::Point(center), radius, 0 - animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 80 - animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 150 - animCount, 0, 30, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 200 - animCount, 0, 30, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 260 - animCount, 0, 60, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 3370 - animCount, 0, 5, color, ring_thickness);

	radius *= 0.9;
	ring_thickness = std::max(int(radius.x / 15), 1);
	cv::ellipse(mat, cv::Point(center), radius, 30 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 110 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 180 + animCount, 0, 30, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 230 + animCount, 0, 10, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 260 + animCount, 0, 10, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 290 + animCount, 0, 60, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 367 + animCount, 0, 5, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
	cv::ellipse(mat, cv::Point(center), radius, 0 + animCount, 0, 50, color, ring_thickness);
}

void drawText(cv::Mat &mat, RECT rect, cv::Scalar color, std::string str, int animCount)
{
	/* Reference: https://github.com/Kazuhito00/object-detection-bbox-art */
	double font_size = std::min((rect.width + rect.height) / 2 * 0.1, 1.0);
	cv::Point drawpoint1(rect.x + rect.width / 2, rect.y + rect.height / 2);
	cv::Point drawpoint2(rect.x + rect.width - rect.width / 10, rect.y + int(rect.height / 10));
	cv::Point drawpoint3(rect.x + rect.width + int(rect.width / 2), rect.y + int(rect.height / 10));
	cv::Point textpoint(drawpoint2.x, drawpoint2.y - int(font_size * 2.0));
	if (drawpoint3.x > mat.cols) {
		drawpoint2 = cv::Point(rect.x + rect.width / 10, rect.y + int(rect.height / 10));
		drawpoint3 = cv::Point(rect.x - rect.width / 2, rect.y + int(rect.height / 10));
		textpoint = cv::Point(drawpoint3.x, drawpoint2.y - int(font_size * 1.5));
	}
	cv::circle(mat, drawpoint1, int(rect.width / 40), color, -1);
	cv::line(mat, drawpoint1, drawpoint2, color, std::max(2, rect.width / 80));
	cv::line(mat, drawpoint2, drawpoint3, color, std::max(2, rect.width / 80));

	cv::putText(mat, str, textpoint, cv::FONT_HERSHEY_DUPLEX, font_size, cv::Scalar(171, 97, 50), 5);
	cv::putText(mat, str, textpoint, cv::FONT_HERSHEY_DUPLEX, font_size, color, 2);
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

	int animCount = 0;
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
				//cv::rectangle(mat, trackedRect, cv::Scalar(255, 0, 0), 2, 1);
				RECT rect;
				rect.x = trackedRect.x;
				rect.y = trackedRect.y;
				rect.width = trackedRect.width;
				rect.height = trackedRect.height;
				drawRing(mat, rect, cv::Scalar(255, 255, 205), animCount);
				drawText(mat, rect, cv::Scalar(207, 161, 69), "TEST aa", animCount);


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
		animCount++;
	}
	for (auto obj : objectList) {
		obj.tracker.release();
	}

	return 0;
}
