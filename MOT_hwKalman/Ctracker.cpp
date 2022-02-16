#include "Ctracker.h"

///
/// \brief CTracker::CTracker
/// Tracker. Manage tracks. Create, remove, update.
/// \param settings
///
CTracker::CTracker(const TrackerSettings& settings)
    :
      m_settings(settings),
      m_nextTrackID(0)
{
    ShortPathCalculator* spcalc = nullptr;
	SPSettings spSettings;
    switch (m_settings.m_matchType)
    {
    case tracking::MatchHungrian:
        spcalc = new SPHungrian(spSettings);
        break;
	//case tracking::MatchBipart:
	//	spcalc = new SPBipart(spSettings);
	//	break;
	}
    assert(spcalc != nullptr);
    m_SPCalculator = std::unique_ptr<ShortPathCalculator>(spcalc);
}

///
/// \brief CTracker::~CTracker
///
CTracker::~CTracker(void)
{
}

///
/// \brief CTracker::Update
/// \param regions
/// \param currFrame
/// \param fps
///
void CTracker::Update(const regions_t& regions,
                      cv::Mat currFrame,
                      float fps)
{
    UpdateTrackingState(regions, currFrame, fps);

    currFrame.copyTo(m_prevFrame);
}

///
/// \brief CTracker::UpdateTrackingState
/// \param regions
/// \param currFrame
/// \param fps
///
void CTracker::UpdateTrackingState(const regions_t& regions,
                                   cv::Mat currFrame,
                                   float fps)
{
    const size_t N = m_tracks.size();	// Tracking objects
    const size_t M = regions.size();	// Detections or regions
	const size_t paddedSize = N + M;

    assignments_t assignment(N, -1); // Assignments regions -> tracks
	assignments_t assignment_all(paddedSize, -1); //
    std::vector<RegionEmbedding> regionEmbeddings;

    if (!m_tracks.empty())			// 如果轨迹不为空，创建代价矩阵
    {
        // Distance matrix between all tracks to all regions
		distMatrix_t costMatrix_NM(N * M, FLT_MAX);
		distMatrix_t costMatrix(paddedSize * paddedSize, FLT_MAX);
        const track_t maxPossibleCost = static_cast<track_t>(currFrame.cols * currFrame.rows);
        track_t maxCost = 0;
        CreateDistaceMatrix(regions, regionEmbeddings, costMatrix_NM, maxPossibleCost, maxCost, currFrame);

		for (int i = 0; i < N; i++)						// i->row
		{
			for (int j = 0; j < M; j++)					// j->col
			{
				costMatrix[i + j * paddedSize] = costMatrix_NM[i + j * N];// 列扫描
			}
		}

		for (int i = 0; i < N; i++) // (i, M+i)
			costMatrix[i + (M + i) * paddedSize] = 8; // costUnmatchedTracks
		for (int j = 0; j < M; j++)	// (N+j, j)
			costMatrix[N + j + j * paddedSize] = 8;	// costUnmatchedDetecions
		for (int i = N; i < paddedSize; i++)
			for (int j = M; j < paddedSize; j++)
				costMatrix[i + j * paddedSize] = 0;
        // Solving assignment problem (shortest paths)
        m_SPCalculator->Solve(costMatrix, paddedSize, paddedSize, assignment_all, maxCost);
		
		// 确认unsignedTracks和unsignedDetections
		for (size_t i = 0; i < N; i++)
		{
			if (assignment_all[i] >= M)	// unsignedTracks
				assignment[i] = -1;
			else
				assignment[i] = assignment_all[i];
		}


        // clean assignment from pairs with large distance
        for (size_t i = 0; i < assignment.size(); i++)
        {
            if (assignment[i] != -1)
            {
				//std::cout << "Track " << i << " match Detection " << assignment[i] << std::endl << std::endl;
#if 0
				if (costMatrix[i + assignment[i] * N] > m_settings.m_distThres)
                {
                    assignment[i] = -1;
                    m_tracks[i]->SkippedFrames()++;
                }
#endif
            }
            else
            {
                // If track have no assigned detect, then increment skipped frames counter.
                m_tracks[i]->SkippedFrames()++;
            }
        }

        // If track didn't get detects long time, remove it.
        for (size_t i = 0; i < m_tracks.size();)
        {
            if (m_tracks[i]->SkippedFrames() > m_settings.m_maximumAllowedSkippedFrames ||
				m_tracks[i]->IsOutOfTheFrame()) //||
				/* m_tracks[i]->IsStaticTimeout(cvRound(fps * (m_settings.m_maxStaticTime - m_settings.m_minStaticTime)))) */
            {
                m_tracks.erase(m_tracks.begin() + i);
                assignment.erase(assignment.begin() + i);
            }
			else
			{
				++i;
			}
        }
    }

    // Search for unassigned detects and start new tracks for them.
    for (size_t i = 0; i < regions.size(); ++i)
    {
        if (find(assignment.begin(), assignment.end(), i) == assignment.end())
        {
            if (regionEmbeddings.empty())	// vector的push_back，并且使用CTrack构造了一个临时对象push进去
                m_tracks.push_back(std::make_unique<CTrack>(regions[i],
                                                            m_settings.m_kalmanType,
                                                            m_settings.m_dt,
                                                            m_settings.m_accelNoiseMag,
                                                            m_settings.m_useAcceleration,	// 使用匀速模型，不是匀加速模型
                                                            m_nextTrackID++,				// 从零开始遍历m_tracks里面的东西
                                                            m_settings.m_filterGoal == tracking::FilterRect,
                                                            m_settings.m_lostTrackType));
            else							// 否则m_tracks里面放的是regionEmbedding里面的东西
                m_tracks.push_back(std::make_unique<CTrack>(regions[i],
                                                            regionEmbeddings[i],
                                                            m_settings.m_kalmanType,
                                                            m_settings.m_dt,
                                                            m_settings.m_accelNoiseMag,
                                                            m_settings.m_useAcceleration,
                                                            m_nextTrackID++,
                                                            m_settings.m_filterGoal == tracking::FilterRect,
                                                            m_settings.m_lostTrackType));
        }
    }

    // Update Kalman Filters state
    const ptrdiff_t stop_i = static_cast<ptrdiff_t>(assignment.size());
#pragma omp parallel for
    for (ptrdiff_t i = 0; i < stop_i; ++i)
    {
        // If track updated less than one time, than filter state is not correct.
        if (assignment[i] != -1) // If we have assigned detect, then update using its coordinates,
        {
            m_tracks[i]->SkippedFrames() = 0;
            if (regionEmbeddings.empty())			// 是否已经初始化，若未初始化
                m_tracks[i]->Update(regions[assignment[i]],
                        true, m_settings.m_maxTraceLength,
                        m_prevFrame, currFrame,		// 是否放弃检测，5*fps后，5s之后，就放弃检测。
                        m_settings.m_useAbandonedDetection ? cvRound(m_settings.m_minStaticTime * fps) : 0);
            else									// 已经初始化过，预测
                m_tracks[i]->Update(regions[assignment[i]], regionEmbeddings[assignment[i]],
                        true, m_settings.m_maxTraceLength,
                        m_prevFrame, currFrame,
                        m_settings.m_useAbandonedDetection ? cvRound(m_settings.m_minStaticTime * fps) : 0);
        }
        else				     // if not continue using predictions 只用预测的部分，没有校正的部分
        {
            m_tracks[i]->Update(CRegion(), false, m_settings.m_maxTraceLength, m_prevFrame, currFrame, 0);
        }
    }
}

///
/// \brief CTracker::CreateDistaceMatrix
/// \param regions
/// \param costMatrix
/// \param maxPossibleCost
/// \param maxCost
///
void CTracker::CreateDistaceMatrix(const regions_t& regions,
                                   std::vector<RegionEmbedding>& regionEmbeddings,
                                   distMatrix_t& costMatrix,
                                   track_t maxPossibleCost,
                                   track_t& maxCost,
                                   cv::Mat currFrame)
{
    const size_t N = m_tracks.size();	// Tracking objects
    maxCost = 0;
#ifdef DEBUG_USE
	std::vector<TrackingObject> m_tracks4debug;
	m_tracks4debug = GetTracks();
	std::cout << "Tracks has " << N << "\t";
	std::cout << "Regions has " << regions.size() << std::endl;
	std::cout << "Track_ID" << "\t";
	for (int i = 0; i < regions.size(); i++)
	{
		std::cout << "\t" << i;
	}
	std::cout << std::endl;
#endif
	for (size_t i = 0; i < N; ++i)		// 轨迹的大小
	{
		const auto& track = m_tracks[i];// CTracker的m_tracks是vector<CTrack>

		// 先调用预测
		if (track->GetFilterObjectSize())
			track->KalmanPredictRect();
		else
			track->KalmanPredictPoint();

#ifdef DEBUG_USE
		const auto& track4debug = m_tracks4debug[i];
		cv::Rect brect = track4debug.m_rrect.boundingRect();
		std::cout << "  " << track4debug.m_ID <<" skip ="<< m_tracks[i]->SkippedFrames()<< "\t\t";
			//<< " ,包围框：(" << brect.x << "," << brect.y << ","
			//<< brect.width << "," << brect.height << ")\t trace_size = " << track4debug.m_trace.size()
			//<< "skipFrame = " << m_tracks[i]->SkippedFrames() << std::endl;
#endif

		// Calc predicted area for track
		cv::Size_<track_t> minRadius;
		if (m_settings.m_minAreaRadiusPix < 0)
		{
			minRadius.width = m_settings.m_minAreaRadiusK * track->LastRegion().m_rrect.size.width;
			minRadius.height = m_settings.m_minAreaRadiusK * track->LastRegion().m_rrect.size.height;
		}
		else
		{
			minRadius.width = m_settings.m_minAreaRadiusPix;
			minRadius.height = m_settings.m_minAreaRadiusPix;
		}
		cv::RotatedRect predictedArea = track->CalcPredictionEllipse(minRadius);

		// Calc distance between track and regions
		for (size_t j = 0; j < regions.size(); ++j)
		{
			const auto& reg = regions[j];
#ifdef DEBUG_USE
			/*std::cout << "\t\t" << "检测" << j << ": 包围框（" << regions[j].m_brect.x << "," << regions[j].m_brect.y << ","
				<< regions[j].m_brect.width << "," << regions[j].m_brect.height << ")" << "\t";*/
			//std::cout << "\t" << j;
#endif
			auto dist = maxPossibleCost;
			if (m_settings.CheckType(m_tracks[i]->LastRegion().m_type, reg.m_type))
			{
				dist = 0;
				size_t ind = 0;
				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistCenters)
				{
#if 1
					/* 计算马氏距离 by zhr*/
					track_t ellipseDist = track->CalcMahalanobisDist(reg.m_rrect);
					dist += ellipseDist;
#ifdef DEBUG_USE
					std::cout <<  ellipseDist << "\t";
#endif
                    //track_t ellipseDist = track->IsInsideArea(reg.m_rrect.center, predictedArea);
                    //if (ellipseDist > 1)//圆内还是圆外，有一个预测圆心的距离。
                    //    dist += m_settings.m_distType[ind];
                    //else
                    //    dist += ellipseDist * m_settings.m_distType[ind];
#else
					dist += m_settings.m_distType[ind] * track->CalcDistCenter(reg);
#endif
				}
				++ind;

				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistRects)
				{
#if 1
                    track_t ellipseDist = track->IsInsideArea(reg.m_rrect.center, predictedArea);
					if (ellipseDist < 1)
					{
						track_t dw = track->WidthDist(reg);
						track_t dh = track->HeightDist(reg);
						dist += m_settings.m_distType[ind] * (1 - (1 - ellipseDist) * (dw + dh) * 0.5f);
					}
					else
					{
						dist += m_settings.m_distType[ind];
					}
					//std::cout << "dist = " << dist << ", ed = " << ellipseDist << ", dw = " << dw << ", dh = " << dh << std::endl;
#else
					dist += m_settings.m_distType[ind] * track->CalcDistRect(reg);
#endif
				}
				++ind;

				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistJaccard)
					dist += m_settings.m_distType[ind] * track->CalcDistJaccard(reg);
				++ind;

				if (m_settings.m_distType[ind] > 0.0f && ind == tracking::DistHist)
                {
                    if (regionEmbeddings.empty())
                        regionEmbeddings.resize(regions.size());
                    dist += m_settings.m_distType[ind] * track->CalcDistHist(reg, regionEmbeddings[j].m_hist, currFrame);
                }
				++ind;
				assert(ind == tracking::DistsCount);
			}

			costMatrix[i + j * N] = dist;
			if (dist > maxCost)
				maxCost = dist;
		}
#ifdef DEBUG_USE
		std::cout << std::endl;
#endif
	}
}
