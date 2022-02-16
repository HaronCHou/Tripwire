#include <opencv2/opencv.hpp>
#include "vibe-background-sequential.h"
#include "Ctracker.h"
#include "defines.h"
#include <iostream>


/*#define SILENT_WORK*/						// �Ƿ�رտ��ӻ�

void detection();
bool InitTracker(cv::Mat frame);
TrackerSettings m_trackerSettings;			 //	���ٵ�������Ϣ 
std::unique_ptr<CTracker> m_tracker;	 	 //	���ٶ���

float m_fps = 1; 
void DetectContour(cv::Mat segmentationMap, regions_t &m_regions);
void Tracking(cv::Mat frame, const regions_t& regions);
void DrawData(cv::Mat frame, int framesCounter, int currTime, cv::Mat fg);
void DrawTrack(cv::Mat frame, int resizeCoeff, const TrackingObject& track, bool drawTrajectory, int framesCounter);
void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha);
void CalcMotionMap(cv::Mat& frame, cv::Mat m_fg);

std::vector<cv::Scalar> m_colors;
void DrawDetect(cv::Mat& segmentationMap, regions_t m_regions);

// ȫ�ֱ����������ڻ��ߵ�ʱ��ʹ��
cv::Mat frame;                  /* Current frame. */

class Line
{
public:
	float x1, y1, x2, y2;
	// count1Ϊ�������(x1,y1)-��x2,y2)�������Ĵ�Խ����
	uchar clockWiseCount = 0, AntiClockWiseCount = 0;
	Line(float _x1, float _y1, float _x2, float _y2, uchar leftCount1 = 0, uchar rightCount2 = 0)
	{
		x1 = _x1; y1 = _y1; x2 = _x2; y2 = _y2;
	}
	Line()
	{
		x1 = 0.0; y1 = 0.0;
		clockWiseCount = 0; AntiClockWiseCount = 0;
	}
};
bool intersection(const Line &l1, const Line &l2);
void checkLineCrosses(std::vector<Line>&Lines, const TrackingObject& track);
void checkLineCross(Line traj, Line &line1);
bool calcVectorAngle(Line traj, Line line1);

// Խ�߼�ⲿ�֣�A,B
Line line1(180, 80, 180, 180);
std::vector<Line> Lines;
void on_mouse(int event, int x, int y, int flags, void* ustc);
void drawLines(cv::Mat frame);

int main(int argc, char** argv) {
	m_colors.push_back(cv::Scalar(255, 0, 0));				 // ��ɫ��ջ
	m_colors.push_back(cv::Scalar(0, 255, 0));
	m_colors.push_back(cv::Scalar(0, 0, 255));
	m_colors.push_back(cv::Scalar(255, 255, 0));
	m_colors.push_back(cv::Scalar(0, 255, 255));
	m_colors.push_back(cv::Scalar(255, 0, 255));
	m_colors.push_back(cv::Scalar(255, 127, 255));
	m_colors.push_back(cv::Scalar(127, 0, 255));
	m_colors.push_back(cv::Scalar(127, 0, 127));

	//Lines.push_back(line1);
	float t1 = (line1.x1 + line1.x2) / 2.0;
	float t2 = (line1.y1 + line1.y2) / 2.0;

	float ux = line1.x2 - line1.x1;
	float uy = line1.y2 - line1.y1; // A->B�ķ���
	float uNorm = sqrt(ux * ux + uy * uy);
	// ˳ʱ�뷽�������
	float leftX = -uy / uNorm * 25; // 25Ϊ����
	float leftY =  ux / uNorm * 25;	// ������㹫ʽ����ֱ����(ux,uy)���߷��򣬴�ֱ����Ϊ(uy,-ux)��(-uy,ux)
				
	// ��ʱ�뷽�����Ҳ�
	float rightX =  uy / uNorm * 25;
	float rightY = -ux / uNorm * 25;

	cv::Point leftArrow((int)leftX, (int)leftY);	// ���ͷ
	cv::Point rightArrow((int)rightX, (int)rightY);
	cv::Point middlePoint((int)t1, (int)t2);		// Ϊ�˻���ͷ��AB���ߵ��ص�

	std::string videoFilename = "G:\\python_based_algo\\Multitarget-tracker\\data\\multiobjec_hard.avi"; 
	// TrackingBugs.mp4 multiobjec_car.avi atrium.mp4  multiobjec_hard.avi  1.avi road.avi
	//std::string videoFilename = "G:\\pic\\testTracker\\testTracker.avi";

	cv::VideoCapture capture(videoFilename);		

	int64 startLoopTime = cv::getTickCount();
	bool m_isDetectorInitialized = false;
	bool m_isTrackerInitialized = false;

	/* Variables. */
	static int framesCounter = 1; /* The current frame number */
	//cv::Mat frame;                  /* Current frame. */
	cv::Mat segmentationMap;        /* Will contain the segmentation map. This is the binary output map. */
	int keyboard = 0;           /* Input from keyboard. Used to stop the program. Enter 'q' to quit. */
								/* Model for ViBe. */
	vibeModel_Sequential_t *model = NULL; /* Model used by ViBe. */
										  /* Read input data. ESC or 'q' for quitting. */
	double freq = cv::getTickFrequency();
	while (/*(char)keyboard != 'q' && (char)keyboard != 27*/ 1) {
		/* Read the current frame. */
		if (!capture.read(frame)) {
			return 0;
		}
		//	detection
		cv::Mat frame_bgr;
		//cvtColor(frame, frame, CV_BGR2GRAY);
		cv::Mat histBuff[20];
		for (int i = 0; i < 20; i++) {
			histBuff[i] = cv::Mat::zeros(frame.rows, frame.cols, CV_8UC1);
		}
		if (framesCounter == 1) {
			// detection ��ʼ��
			segmentationMap = cv::Mat(frame.rows, frame.cols, CV_8UC1);
			model = (vibeModel_Sequential_t*)libvibeModel_Sequential_New();
			libvibeModel_Sequential_AllocInit_8u_C3R(model, frame.data, frame.cols, frame.rows);

			// tracking ��ʼ��
			InitTracker(frame);
		}
		uint8_t *historyImage = (uint8_t*)malloc(2 * frame.cols * frame.rows * sizeof(uint8_t));
		uint8_t *historyBuffer = (uint8_t*)malloc(18 * frame.cols * frame.rows * sizeof(uint8_t));
		int64 t1 = cv::getTickCount();
		/* ViBe: Segmentation and updating. */
		libvibeModel_Sequential_Segmentation_8u_C3R(model, frame.data, segmentationMap.data);
		libvibeModel_Sequential_Update_8u_C3R(model, frame.data, segmentationMap.data/*, historyImage, historyBuffer*/);
		
		// ǰ������
		//cv::medianBlur(segmentationMap, segmentationMap, 3);

		// ������̬ѧ�������ȿ����� 3*3���ٱղ���
		cv::Mat dilateElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
		// cv::dilate(foreground, foreground, dilateElement, cv::Point(-1, -1), 2);
		cv::Mat kernel_close = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(3, 3), cv::Point(-1, -1));
		cv::morphologyEx(segmentationMap, segmentationMap, cv::MORPH_OPEN, dilateElement);
		cv::morphologyEx(segmentationMap, segmentationMap, cv::MORPH_CLOSE, kernel_close);
		
		// tracking
		regions_t m_regions;
		DetectContour(segmentationMap, m_regions);
		DrawDetect(segmentationMap, m_regions);
		cv::imshow("", segmentationMap);

		Tracking(frame, m_regions);

		int64 t2 = cv::getTickCount();
		int currTime = cvRound(1000 * (t2 - t1) / freq);
		DrawData(frame, framesCounter, currTime, segmentationMap);

#if 0
		// �������߻��ư��߻���ȥ
		cv::Point pointA(int(line1.x1), int(line1.y1));
		cv::Point pointB(int(line1.x2), int(line1.y2));
		// ����AB
		cv::line(frame, pointA, pointB, cv::Scalar(0,0,255));
		cv::putText(frame, "A", pointA, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		cv::putText(frame, "B", pointB, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		// ����ͷ��ȥ�����ĵ����ͷ���Ҽ�ͷ
		cv::arrowedLine(frame, middlePoint, middlePoint + leftArrow, cv::Scalar(255, 0, 0), 1,8,0,0.1);
		// �������ĸ���
		cv::putText(frame, std::to_string(Lines[0].rightCount2), middlePoint + leftArrow, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
		cv::putText(frame, std::to_string(Lines[0].leftCount1), middlePoint + rightArrow, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		cv::arrowedLine(frame, middlePoint, middlePoint + rightArrow, cv::Scalar(0, 0, 255),1, 8, 0, 0.1);
#endif
		// ��꽻��İ��߻���
		drawLines(frame);

		cv::imshow("Video", frame);

		// �ڵ�һ֡��ʱ�򣬵ȴ����ߣ�Ȼ��Enter����ִ��
		if (framesCounter == 1)
		{
			// ����������
			cvSetMouseCallback("Video", on_mouse, 0);

			// ��Enter����
			if (cv::waitKey(0) == 27)
			{
				++framesCounter;
				continue;
			}
		}
		cv::waitKey(400);
		++framesCounter;
	}
	return 0;
}
bool InitTracker(cv::Mat frame) {
	bool m_trackerSettingsLoaded = false;
	if (!m_trackerSettingsLoaded)
	{
		m_trackerSettings.SetDistance(tracking::DistCenters);
		m_trackerSettings.m_kalmanType = tracking::KalmanLinear;
		m_trackerSettings.m_filterGoal = tracking::/*FilterRect*/FilterCenter;			// ���ĵ�
		m_trackerSettings.m_lostTrackType = tracking::TrackCSRT;       // Use visual objects tracker for collisions resolving
		m_trackerSettings.m_matchType = tracking::MatchHungrian;
		m_trackerSettings.m_useAcceleration = false;                   // Use constant acceleration motion model
		m_trackerSettings.m_dt = m_trackerSettings.m_useAcceleration ? 0.05f : 1.0f/*0.5*/; // Delta time for Kalman filter
		m_trackerSettings.m_accelNoiseMag = 1.0f;                  // Accel noise manitude for Kalman filter 0.2
		// cost���۾�����޳���ֵ
		m_trackerSettings.m_distThres = 20/*0.95f*/;                    // Distance threshold between region and object on two frames
#if 0
		m_trackerSettings.m_minAreaRadiusPix = frame.rows / 20.f;
#else
		m_trackerSettings.m_minAreaRadiusPix = -1.f;
#endif
		m_trackerSettings.m_minAreaRadiusK = 0.8f;

		m_trackerSettings.m_useAbandonedDetection = true;

		// ���ò��� ���� ɾ���켣�Ĺ���
		if (m_trackerSettings.m_useAbandonedDetection)
		{
			m_trackerSettings.m_minStaticTime = 5;	// m_minStaticTime = 5; ��С��ֹ֡��
			m_trackerSettings.m_maxStaticTime = 10; // ���ֹ֡��  ������������������֡����û�ҵ�ID����ƥ���skipFrames)150,������Ǻܴ� 5sû����
													// ����10û�ҵ��������¿�ʼ��ID
			m_trackerSettings.m_maximumAllowedSkippedFrames = 3;//cvRound(m_trackerSettings.m_minStaticTime * m_fps); // Maximum allowed skipped frames
																// ��໭��10���켣
																/*m_trackerSettings.m_maxTraceLength = 2 * m_trackerSettings.m_maximumAllowedSkippedFrames; */       // Maximum trace length
			m_trackerSettings.m_maxTraceLength = 10;
		}
		else
		{
			m_trackerSettings.m_maximumAllowedSkippedFrames = cvRound(2 * m_fps); // Maximum allowed skipped frames
			m_trackerSettings.m_maxTraceLength = cvRound(4 * m_fps);              // Maximum trace length
		}
	}

	m_tracker = std::make_unique<CTracker>(m_trackerSettings);
	return true;
}

void DrawTrack(cv::Mat frame,
	int resizeCoeff,
	const TrackingObject& track,
	bool drawTrajectory,
	int framesCounter)
{
	auto ResizePoint = [resizeCoeff](const cv::Point& pt) -> cv::Point
	{
		return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
	};

	cv::Scalar color = track.m_isStatic ? cv::Scalar(255, 0, 255) : cv::Scalar(0, 255, 0);
	cv::Point2f rectPoints[4];
	track.m_rrect.points(rectPoints);
	for (int i = 0; i < 4; ++i)
	{
		cv::line(frame, ResizePoint(rectPoints[i]), ResizePoint(rectPoints[(i + 1) % 4]), color);
	}
#if 0
#if 0
	track_t minAreaRadiusPix = frame.rows / 20.f;
#else
	track_t minAreaRadiusPix = -1.f;
#endif
	track_t minAreaRadiusK = 0.5f;
	cv::Size_<track_t> minRadius(minAreaRadiusPix, minAreaRadiusPix);
	if (minAreaRadiusPix < 0)
	{
		minRadius.width = minAreaRadiusK * track.m_rrect.size.width;
		minRadius.height = minAreaRadiusK * track.m_rrect.size.height;
	}

	Point_t d(3.f * track.m_velocity[0], 3.f * track.m_velocity[1]);
	cv::Size2f els(std::max(minRadius.width, fabs(d.x)), std::max(minRadius.height, fabs(d.y)));
	Point_t p1 = track.m_rrect.center;
	Point_t p2(p1.x + d.x, p1.y + d.y);
	float angle = 0;
	Point_t nc = p1;
	Point_t p2_(p2.x - p1.x, p2.y - p1.y);
	if (fabs(p2_.x - p2_.y) > 5) // pix
	{
		if (fabs(p2_.x) > 0.0001f)
		{
			track_t l = std::min(els.width, els.height) / 3;

			track_t p2_l = sqrt(sqr(p2_.x) + sqr(p2_.y));
			nc.x = l * p2_.x / p2_l + p1.x;
			nc.y = l * p2_.y / p2_l + p1.y;

			angle = atan(p2_.y / p2_.x);
		}
		else
		{
			nc.y += d.y / 3;
			angle = CV_PI / 2.f;
		}
	}

	cv::RotatedRect rr(nc, els, 180.f * angle / CV_PI);
	cv::ellipse(frame, rr, cv::Scalar(100, 0, 100), 1);
#endif

	// ���track����ߵĽ���
	checkLineCrosses(Lines, track);

	if (drawTrajectory)
	{
		cv::Scalar cl = m_colors[track.m_ID % m_colors.size()];

		for (size_t j = 0; j < track.m_trace.size() - 1; ++j)
		{
			const TrajectoryPoint& pt1 = track.m_trace.at(j);
			const TrajectoryPoint& pt2 = track.m_trace.at(j + 1);

			// Ϊ�˰�ID��ʾ����
			cv::Rect brect = track.m_rrect.boundingRect();
			std::string label = std::to_string(track.m_ID);
			cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));

#if 0//(CV_VERSION_MAJOR >= 4)
			cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, cv::LINE_AA);
#else
			cv::line(frame, ResizePoint(pt1.m_prediction), ResizePoint(pt2.m_prediction), cl, 1, CV_AA);
#endif
			if (!pt2.m_hasRaw)
			{
#if 0 //(CV_VERSION_MAJOR >= 4)
				cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, cv::LINE_AA);
#else
				cv::circle(frame, ResizePoint(pt2.m_prediction), 4, cl, 1, CV_AA);
#endif
			}
		}
	}

	//cv::Rect brect = track.m_rrect.boundingRect();
	//std::cout << "֡�ţ�" << framesCounter << ", track_ID: " << track.m_ID
	//	<< " ,��Χ��(" << brect.x << "," << brect.y << "," << brect.width << "," << brect.height << ")" << std::endl;
	//m_resultsLog.AddTrack(framesCounter, track.m_ID, brect, track.m_type, track.m_confidence);
	//m_resultsLog.AddRobustTrack(track.m_ID);
}

void DrawData(cv::Mat frame, int framesCounter, int currTime, cv::Mat fg)
{
	std::vector<TrackingObject> m_tracks;
	m_tracks = m_tracker->GetTracks();

	//if (m_showLogs)
		std::cout << "Frame " << framesCounter << ": tracks = " << m_tracks.size() << ", time = " << currTime << std::endl;

	for (const auto& track : m_tracks)
	{
		if (track.m_isStatic)
		{
			DrawTrack(frame, 1, track, false, framesCounter);

			std::string label = "abandoned " + std::to_string(track.m_ID);
			int baseLine = 0;
			cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);

			cv::Rect brect = track.m_rrect.boundingRect();
			if (brect.x < 0)
			{
				brect.width = std::min(brect.width, frame.cols - 1);
				brect.x = 0;
			}
			else if (brect.x + brect.width >= frame.cols)
			{
				brect.x = std::max(0, frame.cols - brect.width - 1);
				brect.width = std::min(brect.width, frame.cols - 1);
			}
			if (brect.y - labelSize.height < 0)
			{
				brect.height = std::min(brect.height, frame.rows - 1);
				brect.y = labelSize.height;
			}
			else if (brect.y + brect.height >= frame.rows)
			{
				brect.y = std::max(0, frame.rows - brect.height - 1);
				brect.height = std::min(brect.height, frame.rows - 1);
			}
			DrawFilledRect(frame, cv::Rect(cv::Point(brect.x, brect.y - labelSize.height), cv::Size(labelSize.width, labelSize.height + baseLine)), cv::Scalar(255, 0, 255), 150);
			cv::putText(frame, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 0));
		}
		else
		{	

			if (track.IsRobust(cvRound(1/*m_fps / 4*/),          // Minimal trajectory size ��������Ż�����
				0.1/*0.7f*/,                        // Minimal ratio raw_trajectory_points / trajectory_lenght
				cv::Size2f(0.1f, 8.0f)))      // Min and max ratio: width / height
				DrawTrack(frame, 1, track, true, framesCounter);
		}
	}
	std::cout << std::endl;
	//CalcMotionMap(frame, fg);
}
void CalcMotionMap(cv::Mat& frame, cv::Mat m_fg)
{
	// Motion map for visualization current detections
	cv::Mat m_motionMap;
	cv::Mat m_normFor;

	if (m_motionMap.size() != frame.size())
		m_motionMap = cv::Mat(frame.size(), CV_32FC1, cv::Scalar(0, 0, 0));

	cv::normalize(m_fg, m_normFor, 255, 0, cv::NORM_MINMAX, m_motionMap.type());

	double alpha = 0.95;
	cv::addWeighted(m_motionMap, alpha, m_normFor, 1 - alpha, 0, m_motionMap);

	const int chans = frame.channels();

	const int height = frame.rows;
#pragma omp parallel for
	for (int y = 0; y < height; ++y)
	{
		uchar* imgPtr = frame.ptr(y);
		float* moPtr = reinterpret_cast<float*>(m_motionMap.ptr(y));
		for (int x = 0; x < frame.cols; ++x)
		{
			for (int ci = chans - 1; ci < chans; ++ci)
			{
				imgPtr[ci] = cv::saturate_cast<uchar>(imgPtr[ci] + moPtr[0]);
			}
			imgPtr += chans;
			++moPtr;
		}
	}
}
void DrawFilledRect(cv::Mat& frame, const cv::Rect& rect, cv::Scalar cl, int alpha)
{
	if (alpha)
	{
		const int alpha_1 = 255 - alpha;
		const int nchans = frame.channels();
		int color[3] = { cv::saturate_cast<int>(cl[0]), cv::saturate_cast<int>(cl[1]), cv::saturate_cast<int>(cl[2]) };
		for (int y = rect.y; y < rect.y + rect.height; ++y)
		{
			uchar* ptr = frame.ptr(y) + nchans * rect.x;
			for (int x = rect.x; x < rect.x + rect.width; ++x)
			{
				for (int i = 0; i < nchans; ++i)
				{
					ptr[i] = cv::saturate_cast<uchar>((alpha_1 * ptr[i] + alpha * color[i]) / 255);
				}
				ptr += nchans;
			}
		}
	}
	else
	{
		cv::rectangle(frame, rect, cl, -1);
	}
}

void Tracking(cv::Mat frame, const regions_t& regions) {
	m_tracker->Update(regions, frame, m_fps);
}

void DetectContour(cv::Mat segmentationMap, regions_t &m_regions) {
	m_regions.clear();
	std::vector<std::vector<cv::Point>> contours;
	std::vector<cv::Vec4i> hierarchy;
#if (CV_VERSION_MAJOR < 4)
	cv::findContours(segmentationMap, contours, hierarchy, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cv::Point());
#else
	cv::findContours(segmentationMap, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE, cv::Point());
#endif
	for (size_t i = 0; i < contours.size(); i++)
	{
		cv::Rect br = cv::boundingRect(contours[i]);

		//if (br.width >= m_minObjectSize.width &&
		//	br.height >= m_minObjectSize.height)
		if (br.width >= 5 &&
			br.height >= 5)
		{
			if (false/*m_useRotatedRect*/)
			{
				cv::RotatedRect rr = cv::minAreaRect(contours[i]);
				m_regions.push_back(CRegion(rr));
			}
			else
			{
				m_regions.push_back(CRegion(br));
			}
		}
	}
	return;
}
void DrawDetect(cv::Mat& segmentationMap, regions_t m_regions) {
	int resizeCoeff = 1;
	auto ResizePoint = [resizeCoeff](const cv::Point& pt) -> cv::Point
	{
		return cv::Point(resizeCoeff * pt.x, resizeCoeff * pt.y);
	};
	for (int i = 0; i < m_regions.size(); i++)
	{
		cv::Point2f rectPoints[4];
		m_regions[i].m_rrect.points(rectPoints);
		for (int i = 0; i < 4; ++i)
		{
			cv::line(segmentationMap, ResizePoint(rectPoints[i]), ResizePoint(rectPoints[(i + 1) % 4]), cv::Scalar(255,255,255));
		}
		cv::Rect brect = m_regions[i].m_rrect.boundingRect();
		std::string label = std::to_string(i);
		cv::putText(segmentationMap, label, brect.tl(), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
	}
}


// ����Խ�߼��
void checkLineCrosses(std::vector<Line>&Lines, const TrackingObject& track)
{
	for (int i = 0; i < Lines.size();i++)
	{
		int count = track.m_trace.size();
		if (count <= 1) continue;
		// ���Ĺ켣��
		const TrajectoryPoint& pt1 = track.m_trace.at(count - 1);	// ��ǰλ��
		const TrajectoryPoint& pt2 = track.m_trace.at(count - 2);	// ��һ֡λ��
		Line traj(pt1.m_prediction.x, pt1.m_prediction.y, pt2.m_prediction.x, pt2.m_prediction.y);
		// �켣��ÿ�����ߵĽ�����֤
		checkLineCross(traj, Lines[i]);
	}
}
// ����н�
bool calcVectorAngle(Line traj, Line line1)
{
	float ux = traj.x2 - traj.x1;
	float uy = traj.y2 - traj.y1;	// �켣����
	float vx = line1.x2 - line1.x1;
	float vy = line1.y2 - line1.y1;	// ���ߵ�λ��
	// �켣 �� ���� < 0���켣�ڰ��ߵ���ʱ�룬���
	bool isLineLeft = ux*vy - uy*vx < 0;	// ���������ĵ��
	// ux*vy - uy*vx < 0, u �� v == traj �� line < 0 ��˵���켣�ڰ��ߵ����
	return isLineLeft;
}

void checkLineCross(Line traj, Line &line1)
{
	bool isIntersect = intersection(traj, line1);
	if (isIntersect)		// ����켣����ߵļн�
	{
		// 
		bool isLineLeft = calcVectorAngle(traj, line1);
		if (isLineLeft)
			line1.clockWiseCount += 1;	// 
		else {
			line1.AntiClockWiseCount += 1;
		}
	}
}
bool intersection(const Line &l1, const Line &l2)
{
	//�����ų�ʵ��
	if ((l1.x1 > l1.x2 ? l1.x1 : l1.x2) < (l2.x1 < l2.x2 ? l2.x1 : l2.x2) ||
		(l1.y1 > l1.y2 ? l1.y1 : l1.y2) < (l2.y1 < l2.y2 ? l2.y1 : l2.y2) ||
		(l2.x1 > l2.x2 ? l2.x1 : l2.x2) < (l1.x1 < l1.x2 ? l1.x1 : l1.x2) ||
		(l2.y1 > l2.y2 ? l2.y1 : l2.y2) < (l1.y1 < l1.y2 ? l1.y1 : l1.y2))
	{
		return false;
	}
	//����ʵ��
	if ((((l1.x1 - l2.x1)*(l2.y2 - l2.y1) - (l1.y1 - l2.y1)*(l2.x2 - l2.x1))*
		((l1.x2 - l2.x1)*(l2.y2 - l2.y1) - (l1.y2 - l2.y1)*(l2.x2 - l2.x1))) > 0 ||
		(((l2.x1 - l1.x1)*(l1.y2 - l1.y1) - (l2.y1 - l1.y1)*(l1.x2 - l1.x1))*
			((l2.x2 - l1.x1)*(l1.y2 - l1.y1) - (l2.y2 - l1.y1)*(l1.x2 - l1.x1))) > 0)
	{
		return false;
	}
	return true;
}
void on_mouse(int event, int x, int y, int flags, void* ustc)
{
	static cv::Point pre_pt;		// ��һ����
	static cv::Point cur_pt;		// �ڶ�����
	char temp_1[20];
	// ���Ҫ��ͼƬ������λ����Ϊ��ʼ�㣬�������Ͳ���Ҫ��
	//pre_pt=Point(-1,-1);
	//cur_pt=Point(-1,-1);
	if (event == CV_EVENT_LBUTTONDOWN)
	{
		pre_pt = cv::Point(x, y);	// ��ǰ����λ��(x,y)
		sprintf(temp_1,"x:%d,y:%d",x,y);
		//xiaolei=Rect(x1,y1,0,0);
		cv::putText(frame, temp_1, cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255));
		circle(frame, pre_pt, 3, cvScalar(255, 0, 0), CV_FILLED, CV_AA, 0);
		imshow("Video", frame);
	}
	else if (event == CV_EVENT_MOUSEMOVE && (flags & CV_EVENT_FLAG_LBUTTON))
	{
		cur_pt = cv::Point(x, y);
		//sprintf(temp_1, "x:%d,y:%d", x, y);
		//xiaolei=Rect(x1,y1,0,0);
		//cv::putText(frame, temp_1, cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 255, 255));
		//line(frame, pre_pt, cur_pt, cvScalar(0, 255, 0), 1, CV_AA, 0);
		imshow("Video", frame);
	}
	else if (event == CV_EVENT_LBUTTONUP)
	{
		cur_pt = cv::Point(x, y);
		sprintf(temp_1, "x:%d,y:%d", x, y);
		//xiaolei=Rect(x1,y1,0,0);
		cv::putText(frame, temp_1, cv::Point(x, y), CV_FONT_HERSHEY_SIMPLEX, 0.4, cv::Scalar(0, 255, 255));
		circle(frame, cur_pt, 3, cvScalar(255, 0, 0), CV_FILLED, CV_AA, 0);
		line(frame, pre_pt, cur_pt, cvScalar(0, 255, 0), 1, CV_AA, 0);
		imshow("Video", frame);

		// push����
		// Խ�߼�ⲿ�֣�A,B
		Line tempLine((float)pre_pt.x, (float)pre_pt.y, (float)x, (float)y);
		Lines.push_back(tempLine);
	}
}
void drawLines(cv::Mat frame)
{
	for (size_t i = 0; i < Lines.size(); i++)
	{
		cv::Point pointA(int(Lines[i].x1), int(Lines[i].y1));
		cv::Point pointB(int(Lines[i].x2), int(Lines[i].y2));

		float t1 = (Lines[i].x1 + Lines[i].x2) / 2.0;
		float t2 = (Lines[i].y1 + Lines[i].y2) / 2.0;

		float ux = Lines[i].x2 - Lines[i].x1;
		float uy = Lines[i].y2 - Lines[i].y1; // A->B�ķ���
		float uNorm = sqrt(ux * ux + uy * uy);
		// ˳ʱ�뷽�������
		float AnticlockWiseX = -uy / uNorm * 25; // 25Ϊ����
		float AnticlockWiseY = ux / uNorm * 25;	// ������㹫ʽ����ֱ����(ux,uy)���߷��򣬴�ֱ����Ϊ(uy,-ux)��(-uy,ux)
										// ��ʱ�뷽�����Ҳ�
		float clockWiseX = uy / uNorm * 25;
		float clockWiseY = -ux / uNorm * 25;

		cv::Point AnticlockWiseArrow((int)AnticlockWiseX, (int)AnticlockWiseY);	// ���ͷ
		cv::Point clockWiseArrow((int)clockWiseX, (int)clockWiseY);
		cv::Point middlePoint((int)t1, (int)t2);		// Ϊ�˻���ͷ��AB���ߵ��ص�

														// ����AB
		cv::line(frame, pointA, pointB, cv::Scalar(0, 0, 255));
		cv::putText(frame, "A", pointA, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		cv::putText(frame, "B", pointB, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		// ����ͷ��ȥ�����ĵ����ͷ���Ҽ�ͷ
		cv::arrowedLine(frame, middlePoint, middlePoint + AnticlockWiseArrow, cv::Scalar(255, 0, 0), 1, 8, 0, 0.1);
		// �������ĸ�����bgr. ��ʱ�룬arrow=(-y,x) -y*y-x*x<0
		cv::putText(frame, std::to_string(Lines[i].AntiClockWiseCount), middlePoint + AnticlockWiseArrow, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 0, 0));
		// ��AB�ģ�red,<0��Ϊ��-y,x��
		cv::putText(frame, std::to_string(Lines[i].clockWiseCount), middlePoint + clockWiseArrow, cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(0, 0, 255));
		cv::arrowedLine(frame, middlePoint, middlePoint + clockWiseArrow, cv::Scalar(0, 0, 255), 1, 8, 0, 0.1);
	}
}