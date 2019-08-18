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

#ifndef CCNF_PATCH_EXPERT_H
#define CCNF_PATCH_EXPERT_H

#include <opencv2/core/core.hpp>

#include <map>
#include <vector>

namespace LandmarkDetector
{

//===========================================================================
/** 
	A single Neuron response
*/
class CCNF_neuron{

public:

	// Type of patch (0=raw,1=grad,3=depth, other types besides raw are not actually used now)
	int     neuron_type; 

	// scaling of weights (needed as the energy of neuron might not be 1) 
	double  norm_weights; 

	// Weight bias
	double  bias;

	// Neural weights
	cv::Mat_<float> weights; 

	// can have neural weight dfts that are calculated on the go as needed, this allows us not to recompute
	// the dft of the template each time, improving the speed of tracking
	std::map<int, cv::Mat_<double> > weights_dfts;

	// the alpha associated with the neuron
	double alpha; 

	// Default constructor
	CCNF_neuron(){;}

	// Copy constructor
	CCNF_neuron(const CCNF_neuron& other);

	void Read(std::ifstream &stream);
	// The im_dft, integral_img, and integral_img_sq are precomputed images for convolution speedups (they get set if passed in empty values)
	void Response(const cv::Mat_<float> &im, cv::Mat_<double> &im_dft, cv::Mat &integral_img, cv::Mat &integral_img_sq, cv::Mat_<float> &resp);

};

//===========================================================================
/**
A CCNF patch expert
*/
class CCNF_patch_expert{
public:
    
	// Width and height of the patch expert support region
	int width;
	int height;             
    
	// Collection of neurons for this patch expert
	std::vector<CCNF_neuron> neurons;

	// Information about the vertex features (association potentials)
	std::vector<int>				window_sizes;
	std::vector<cv::Mat_<float> >	Sigmas;
	std::vector<double>				betas;

	// Combined weight matrix from each neuron
	cv::Mat_<float> weight_matrix;

	// How confident we are in the patch
	double   patch_confidence;

	// Default constructor
	CCNF_patch_expert(){;}

	// Copy constructor		
	CCNF_patch_expert(const CCNF_patch_expert& other);

	void Read(std::ifstream &stream, std::vector<int> window_sizes, std::vector<std::vector<cv::Mat_<float> > > sigma_components);

	// actual work (can pass in an image and a potential depth image, if the CCNF is trained with depth)
	void Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response);

	void ResponseOpenBlas(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response, cv::Mat_<float> &im2col_prealloc);

	// Helper function to compute relevant sigmas
	void ComputeSigmas(std::vector<cv::Mat_<float> > sigma_components, int window_size);
	
};
  //===========================================================================
}
#endif // CCNF_PATCH_EXPERT_H
