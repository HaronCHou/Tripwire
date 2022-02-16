#pragma once

#include <vector>
#include <string>
#include <map>
#include <opencv2/opencv.hpp>
#include "object_types.h"

#define DEBUG_USE

// ---------------------------------------------------------------------------
//
// ---------------------------------------------------------------------------
typedef float track_t;
typedef cv::Point_<track_t> Point_t;
#define El_t CV_32F
#define Mat_t CV_32FC

// #define USE_OCV_KCF 需要opencv_contrib里面的KCF才行

typedef std::vector<int> assignments_t;
typedef std::vector<track_t> distMatrix_t;

///
/// \brief config_t
///
typedef std::multimap<std::string, std::string> config_t;

///
/// \brief The CRegion class
///
class CRegion
{
public:
	CRegion() = default;

    CRegion(const cv::Rect& rect)
        : m_brect(rect)
    {
        B2RRect();		// bounding box to rotated box 包围框还是旋转框
    }

    CRegion(const cv::RotatedRect& rrect)
        : m_rrect(rrect)
    {
        R2BRect();
    }

    CRegion(const cv::Rect& rect, objtype_t type, float confidence)
        : m_type(type), m_brect(rect), m_confidence(confidence)
    {
        B2RRect();
    }

	objtype_t m_type = bad_type;		// 数据结构：包围框还是旋转框；int 类型
    cv::RotatedRect m_rrect;
    cv::Rect m_brect;
	float m_confidence = -1;

private:
    ///
    /// \brief R2BRect
    /// \return
    ///
    cv::Rect R2BRect()
    {
        m_brect = m_rrect.boundingRect();
        return m_brect;
    }
    ///
    /// \brief B2RRect
    /// \return
    ///
    cv::RotatedRect B2RRect()
    {
        m_rrect = cv::RotatedRect(m_brect.tl(), cv::Point2f(static_cast<float>(m_brect.x + m_brect.width), static_cast<float>(m_brect.y)), m_brect.br());
        return m_rrect;
    }
};

typedef std::vector<CRegion> regions_t;

///
///
///
namespace tracking
{
///
/// \brief The Detectors enum
///
enum Detectors
{
    Motion_VIBE = 0,			// 枚举是默认1,2...一直编号下去
    Motion_MOG,
    Motion_GMG,
    Motion_CNT,
    Motion_SuBSENSE,
    Motion_LOBSTER,
    Motion_MOG2,
    Face_HAAR,
    Pedestrian_HOG,
    Pedestrian_C4,
	Yolo_Darknet,
    Yolo_TensorRT,
    DNN_OCV,
    DetectorsCount
};

///
/// \brief The DistType enum
///
enum DistType
{
    DistCenters,   // Euclidean distance between centers, pixels
    DistRects,     // Euclidean distance between bounding rectangles, pixels
    DistJaccard,   // Intersection over Union, IoU, [0, 1]
	DistHist,      // Bhatacharia distance between histograms, [0, 1]
	DistsCount
};

///
/// \brief The FilterGoal enum
///
enum FilterGoal
{
    FilterCenter,
    FilterRect,
    FiltersCount
};

///
/// \brief The KalmanType enum
///
enum KalmanType
{
    KalmanLinear,
    KalmanUnscented,
    KalmanAugmentedUnscented,
    KalmanCount
};

///
/// \brief The MatchType enum
///
enum MatchType
{
    MatchHungrian,
    MatchBipart,
    MatchCount
};

///
/// \brief The LostTrackType enum
///
enum LostTrackType
{
    TrackNone,
    TrackKCF,
    TrackMIL,
    TrackMedianFlow,
    TrackGOTURN,
    TrackMOSSE,
    TrackCSRT,
    TrackDAT,
    TrackSTAPLE,
    TrackLDES,
    SingleTracksCount
};
}
