///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis, all rights reserved.
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

//  Parameters of the Face analyser
#ifndef RECORDER_OPENFACE_PARAM_H
#define RECORDER_OPENFACE_PARAM_H

#include <vector>
#include <opencv2/core/core.hpp>

namespace Utilities
{

	class RecorderOpenFaceParameters
	{

	public:

		// Constructors
		RecorderOpenFaceParameters(std::vector<std::string> &arguments, bool sequence, bool is_from_webcam, float fx = -1, float fy = -1, float cx = -1, float cy = -1, double fps_vid_out = 30);
		RecorderOpenFaceParameters(bool sequence, bool is_from_webcam, bool output_2D_landmarks, bool output_3D_landmarks,
			bool output_model_params, bool output_pose, bool output_AUs, bool output_gaze, bool output_hog, bool output_tracked,
			bool output_aligned_faces, bool record_bad = true, float fx = -1, float fy = -1, float cx = -1, float cy = -1, double fps_vid_out = 30);

		bool isSequence() const { return is_sequence; }
		bool isFromWebcam() const { return is_from_webcam; }
		bool output2DLandmarks() const { return output_2D_landmarks; }
		bool output3DLandmarks() const { return output_3D_landmarks; }
		bool outputPDMParams() const { return output_model_params; }
		bool outputPose() const { return output_pose; }
		bool outputAUs() const { return output_AUs; }
		bool outputGaze() const { return output_gaze; }
		bool outputHOG() const { return output_hog; }
		bool outputTracked() const { return output_tracked; }
		bool outputAlignedFaces() const { return output_aligned_faces; }
		std::string outputCodec() const { return output_codec; }
		std::string imageFormatAligned() const { return image_format_aligned; }
		std::string imageFormatVisualization() const { return image_format_visualization; }
		double outputFps() const { return fps_vid_out; }

		bool outputBadAligned() const { return record_aligned_bad; }

		float getFx() const { return fx; }
		float getFy() const { return fy; }
		float getCx() const { return cx; }
		float getCy() const { return cy; }

		void setOutputAUs(bool output_AUs) { this->output_AUs = output_AUs; }
		void setOutputGaze(bool output_gaze) { this->output_gaze = output_gaze; }

	private:
		
		// If we are recording results from a sequence each row refers to a frame, if we are recording an image each row is a face
		bool is_sequence;
		// If the data is coming from a webcam
		bool is_from_webcam;

		// Keep track of what we are recording
		bool output_2D_landmarks;
		bool output_3D_landmarks;
		bool output_model_params;
		bool output_pose;
		bool output_AUs;
		bool output_gaze;
		bool output_hog;
		bool output_tracked;
		bool output_aligned_faces;
		
		// Should the algined faces be recorded even if the detection failed (blank images)
		bool record_aligned_bad;

		// Some video recording parameters
		std::string output_codec;
		double fps_vid_out;

		// Image recording parameters
		std::string image_format_aligned;
		std::string image_format_visualization;

		// Camera parameters for recording in the meta file;
		float fx, fy, cx, cy;

	};

}

#endif // RECORDER_OPENFACE_PARAM_H
