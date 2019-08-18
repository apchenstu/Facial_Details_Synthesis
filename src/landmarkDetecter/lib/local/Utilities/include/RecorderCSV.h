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

#ifndef RECORDER_CSV_H
#define RECORDER_CSV_H

// System includes
#include <fstream>
#include <sstream>
#include <vector>

// OpenCV includes
#include <opencv2/core/core.hpp>

namespace Utilities
{

	//===========================================================================
	/**
	A class for recording CSV file from OpenFace
	*/
	class RecorderCSV {

	public:

		// The constructor for the recorder, need to specify if we are recording a sequence or not
		RecorderCSV();

		// Opening the file and preparing the header for it
		bool Open(std::string output_file_name, bool is_sequence, bool output_2D_landmarks, bool output_3D_landmarks, bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze,
			int num_face_landmarks, int num_model_modes, int num_eye_landmarks, const std::vector<std::string>& au_names_class, const std::vector<std::string>& au_names_reg);

		bool isOpen() const { return output_file.is_open(); }

		// Closing the file and cleaning up
		void Close();

		void WriteLine(int face_id, int frame_num, double time_stamp, bool landmark_detection_success, double landmark_confidence,
			const cv::Mat_<float>& landmarks_2D, const cv::Mat_<float>& landmarks_3D, const cv::Mat_<float>& pdm_model_params, const cv::Vec6f& rigid_shape_params, cv::Vec6f& pose_estimate,
			const cv::Point3f& gazeDirection0, const cv::Point3f& gazeDirection1, const cv::Vec2f& gaze_angle, const std::vector<cv::Point2f>& eye_landmarks2d, const std::vector<cv::Point3f>& eye_landmarks3d,
			const std::vector<std::pair<std::string, double> >& au_intensities, const std::vector<std::pair<std::string, double> >& au_occurences);

	private:

		// Blocking copy and move, as it doesn't make sense to read to write to the same file
		RecorderCSV & operator= (const RecorderCSV& other);
		RecorderCSV & operator= (const RecorderCSV&& other);
		RecorderCSV(const RecorderCSV&& other);
		RecorderCSV(const RecorderCSV& other);

		// The actual output file stream that will be written
		std::ofstream output_file;

		// If we are recording results from a sequence each row refers to a frame, if we are recording an image each row is a face
		bool is_sequence;

		// Keep track of what we are recording
		bool output_2D_landmarks;
		bool output_3D_landmarks;
		bool output_model_params;
		bool output_pose;
		bool output_AUs;
		bool output_gaze;

		std::vector<std::string> au_names_class;
		std::vector<std::string> au_names_reg;

	};
}
#endif // RECORDER_CSV_H