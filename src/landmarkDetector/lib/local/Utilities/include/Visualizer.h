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

#ifndef VISUALIZER_H
#define VISUALIZER_H

// System includes
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace Utilities
{

	//===========================================================================
	/**
	A class for recording data processed by OpenFace (facial landmarks, head pose, facial action units, aligned face, HOG features, and tracked video
	*/
	class Visualizer {

	public:

		// The constructor for the visualizer that specifies what to visualize
		Visualizer(std::vector<std::string> arguments);
		Visualizer(bool vis_track, bool vis_hog, bool vis_align, bool vis_aus);

		// Adding observations to the visualizer
		
		// Pose related observations
		void SetImage(const cv::Mat& canvas, float fx, float fy, float cx, float cy);

		// All observations relevant to facial landmarks (optional visibilities parameter to not display all landmarks)
		void SetObservationLandmarks(const cv::Mat_<float>& landmarks_2D, double confidence, const cv::Mat_<int>& visibilities = cv::Mat_<int>());

		// Pose related observations
		void SetObservationPose(const cv::Vec6f& pose, double confidence);
		
		void SetObservationActionUnits(const std::vector<std::pair<std::string, double> >& au_intensities, const std::vector<std::pair<std::string, double> >& au_occurences);

		// Gaze related observations
		void SetObservationGaze(const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1, const std::vector<cv::Point2f>& eye_landmarks, const std::vector<cv::Point3f>& eye_landmarks3d, double confidence);

		// Face alignment related observations
		void SetObservationFaceAlign(const cv::Mat& aligned_face);

		// HOG feature related observations
		void SetObservationHOG(const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows);

		void SetFps(double fps);

		// Return key-press that could have resulted in the open windows
		char ShowObservation();

		cv::Mat GetVisImage();
		cv::Mat GetHOGVis();

		// Keeping track of what we're visualizing
		bool vis_track;
		bool vis_hog;
		bool vis_align;
		bool vis_aus;

		// Can be adjusted to show less confident frames
		double visualisation_boundary = 0.4;

	private:

		// Temporary variables for visualization
		cv::Mat captured_image; // out canvas
		cv::Mat tracked_image;
		cv::Mat hog_image;
		cv::Mat aligned_face_image;
		cv::Mat action_units_image;

		// Useful for drawing 3d
		float fx, fy, cx, cy;

	};
}
#endif