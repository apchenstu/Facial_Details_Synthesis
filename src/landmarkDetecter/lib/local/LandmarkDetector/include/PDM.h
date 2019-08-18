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

#ifndef PDM_H
#define PDM_H

// OpenCV includes
#include <opencv2/core/core.hpp>

#include "LandmarkDetectorParameters.h"

namespace LandmarkDetector
{
//===========================================================================
// A linear 3D Point Distribution Model (constructed using Non-Rigid structure from motion or PCA)
// Only describes the model but does not contain an instance of it (no local or global parameters are stored here)
// Contains the utility functions to help manipulate the model

class PDM{
	public:    
    
		// The 3D mean shape vector of the PDM [x1,..,xn,y1,...yn,z1,...,zn]
		cv::Mat_<float> mean_shape;	
  
		// Principal components or variation bases of the model, 
		cv::Mat_<float> princ_comp;

		// Eigenvalues (variances) corresponding to the bases
		cv::Mat_<float> eigen_values;

		PDM(){;}
		
		// A copy constructor
		PDM(const PDM& other);
			
		bool Read(std::string location);

		// Number of vertices
		inline int NumberOfPoints() const {return mean_shape.rows/3;}
		
		// Listing the number of modes of variation
		inline int NumberOfModes() const {return princ_comp.cols;}

		void Clamp(cv::Mat_<float>& params_local, cv::Vec6f& params_global, const FaceModelParameters& params);

		// Compute shape in object space (3D)
		void CalcShape3D(cv::Mat_<float>& out_shape, const cv::Mat_<float>& params_local) const;

		// Compute shape in image space (2D)
		void CalcShape2D(cv::Mat_<float>& out_shape, const cv::Mat_<float>& params_local, const cv::Vec6f& params_global) const;
    
		// provided the bounding box of a face and the local parameters (with optional rotation), generates the global parameters that can generate the face with the provided bounding box
		void CalcParams(cv::Vec6f& out_params_global, const cv::Rect_<float>& bounding_box, const cv::Mat_<float>& params_local, const cv::Vec3f rotation = cv::Vec3f(0.0f));

		// Provided the landmark location compute global and local parameters best fitting it (can provide optional rotation for potentially better results)
		void CalcParams(cv::Vec6f& out_params_global, cv::Mat_<float>& out_params_local, const cv::Mat_<float>& landmark_locations, const cv::Vec3f rotation = cv::Vec3f(0.0f));

		// provided the model parameters, compute the bounding box of a face
		void CalcBoundingBox(cv::Rect_<float>& out_bounding_box, const cv::Vec6f& params_global, const cv::Mat_<float>& params_local);

		// Helpers for computing Jacobians, and Jacobians with the weight matrix
		void ComputeRigidJacobian(const cv::Mat_<float>& params_local, const cv::Vec6f& params_global, cv::Mat_<float> &Jacob, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w);
		void ComputeJacobian(const cv::Mat_<float>& params_local, const cv::Vec6f& params_global, cv::Mat_<float> &Jacobian, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w);

		// Given the current parameters, and the computed delta_p compute the updated parameters
		void UpdateModelParameters(const cv::Mat_<float>& delta_p, cv::Mat_<float>& params_local, cv::Vec6f& params_global);

	private:
		// Helper utilities
		static void Orthonormalise(cv::Matx33f &R);
  };
  //===========================================================================
}
#endif // PDM_H
