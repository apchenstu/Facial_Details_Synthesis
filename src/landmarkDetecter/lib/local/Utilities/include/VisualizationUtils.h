///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis all rights reserved.
//
// ACADEMIC OR NON-PROFIT ORGANIZATION NONCOMMERCIAL RESEARCH USE ONLY
//
// BY USING OR DOWNLOADING THE SOFTWARE, YOU ARE AGREEING TO THE TERMS OF THIS LICENSE AGREEMENT.  
// IF YOU DO NOT AGREE WITH THESE TERMS, YOU MAY NOT USE OR DOWNLOAD THE SOFTWARE.
//
// License can be found in OpenFace-license.txt
//
//     * Any publications arising from the use of this software, including but
//       not limited to academic journal and conference publications, technical
//       reports and manuals, must cite at least one of the following works:
//
//       OpenFace 2.0: Facial Behavior Analysis Toolkit
//       Tadas Baltrušaitis, Amir Zadeh, Yao Chong Lim, and Louis-Philippe Morency
//       in IEEE International Conference on Automatic Face and Gesture Recognition, 2018  
//
//       Convolutional experts constrained local model for facial landmark detection.
//       A. Zadeh, T. Baltrušaitis, and Louis-Philippe Morency,
//       in Computer Vision and Pattern Recognition Workshops, 2017.    
//
//       Rendering of Eyes for Eye-Shape Registration and Gaze Estimation
//       Erroll Wood, Tadas Baltrušaitis, Xucong Zhang, Yusuke Sugano, Peter Robinson, and Andreas Bulling 
//       in IEEE International. Conference on Computer Vision (ICCV),  2015 
//
//       Cross-dataset learning and person-specific normalisation for automatic Action Unit detection
//       Tadas Baltrušaitis, Marwa Mahmoud, and Peter Robinson 
//       in Facial Expression Recognition and Analysis Challenge, 
//       IEEE International Conference on Automatic Face and Gesture Recognition, 2015 
//
///////////////////////////////////////////////////////////////////////////////

#ifndef VISUALIZATION_UTILS_H
#define VISUALIZATION_UTILS_H
#include <stdafx_ut.h>

namespace Utilities
{

	// Drawing a bounding box around the face in an image
	void DrawBox(cv::Mat image, cv::Vec6f pose, cv::Scalar color, int thickness, float fx, float fy, float cx, float cy);
	void DrawBox(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& lines, cv::Mat image, cv::Scalar color, int thickness);

	// Computing a bounding box to be drawn
	std::vector<std::pair<cv::Point2f, cv::Point2f>> CalculateBox(cv::Vec6f pose, float fx, float fy, float cx, float cy);

    void Visualise_FHOG(const cv::Mat_<double>& descriptor, int num_rows, int num_cols, cv::Mat& visualisation);

	class FpsTracker
	{
	public:
		
		double history_length;

		void AddFrame();

		double GetFPS();

		FpsTracker();

	private:
		std::queue<double> frame_times;

		void DiscardOldFrames();

	};

}
#endif // VISUALIZATION_UTILS_H