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


#ifndef SVR_PATCH_EXPERT_H
#define SVR_PATCH_EXPERT_H

// system includes
#include <map>

// OpenCV includes
#include <opencv2/core/core.hpp>

namespace LandmarkDetector
{
  //===========================================================================
  /** 
      The classes describing the SVR patch experts
  */

class SVR_patch_expert{
	public:

		// Type of data the patch expert operated on (0=raw, 1=grad)
		int     type;					

		// Logistic regression slope
		double  scaling;
		
		// Logistic regression bias
		double  bias;

		// Support vector regression weights
		cv::Mat_<float> weights;

		// Discrete Fourier Transform of SVR weights, precalculated for speed (at different window sizes)
		std::map<int, cv::Mat_<double> > weights_dfts;

		// Confidence of the current patch expert (used for NU_RLMS optimisation)
		double  confidence;

		SVR_patch_expert(){;}
		
		// A copy constructor
		SVR_patch_expert(const SVR_patch_expert& other);

		// Reading in the patch expert
		void Read(std::ifstream &stream);

		// The actual response computation from intensity or depth (for CLM-Z)
		void Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response);
		void ResponseDepth(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response);

};
//===========================================================================
/**
    A Multi-patch Expert that can include different patch types. Raw pixel values or image gradients
*/
class Multi_SVR_patch_expert{
	public:
		
		// Width and height of the patch expert support area
		int width;
		int height;						

		// Vector of all of the patch experts (different modalities) for this particular Multi patch expert
		std::vector<SVR_patch_expert> svr_patch_experts;	

		// Default constructor
		Multi_SVR_patch_expert(){;}
	
		// Copy constructor				
		Multi_SVR_patch_expert(const Multi_SVR_patch_expert& other);

		void Read(std::ifstream &stream);

		// actual response computation from intensity of depth (for CLM-Z)
		void Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response);
		void ResponseDepth(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response);

};
}
#endif // SVR_PATCH_EXPERT_H
