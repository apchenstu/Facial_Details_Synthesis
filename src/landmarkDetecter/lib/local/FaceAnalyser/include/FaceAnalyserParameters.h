///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Carnegie Mellon University and University of Cambridge,
// all rights reserved.
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
#ifndef FACE_ANALYSER_PARAM_H
#define FACE_ANALYSER_PARAM_H

#include <stdafx_fa.h>

namespace FaceAnalysis
{

struct FaceAnalyserParameters
{
public:
	// Constructors
	FaceAnalyserParameters();
	FaceAnalyserParameters(std::string root_exe);
	FaceAnalyserParameters(std::vector<std::string> &arguments);

	// These are the parameters of training and will not change and are fixed
	const double sim_scale_au = 0.7;
	const int sim_size_au = 112;

	// Should the output aligned faces be grayscale
	bool grayscale;

	// Use getters and setters for these as they might need to reload models and make sure the scale and size ratio makes sense
	void setAlignedOutput(int output_size, double scale=-1, bool masked = true);
	// This will also change the model location
	void OptimizeForVideos();
	void OptimizeForImages();

	bool getAlignMask() const { return sim_align_face_mask; }
	double getSimScaleOut() const { return sim_scale_out; }
	int getSimSizeOut() const { return sim_size_out; }
	bool getDynamic() const { return dynamic; }
	std::string getModelLoc() const { return std::string(model_location); }
	std::vector<cv::Vec3d> getOrientationBins() const { return std::vector<cv::Vec3d>(orientation_bins); }

private:

	void init();

	// Aligned face output size
	double sim_scale_out;
	int sim_size_out;

	// Should aligned face be masked out from background
	bool sim_align_face_mask;

	// Should a video stream be assumed
	bool dynamic;

	// Where to load the models from
	std::string model_location;
	// The location of the executable
	fs::path root;

	std::vector<cv::Vec3d> orientation_bins;

};

}

#endif // FACE_ANALYSER_PARAM_H
