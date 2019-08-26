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

#include "stdafx.h"

#include "Patch_experts.h"

#include "RotationHelpers.h"

// Math includes
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;

// A copy constructor

Patch_experts::Patch_experts(const Patch_experts& other) : patch_scaling(other.patch_scaling), centers(other.centers), svr_expert_intensity(other.svr_expert_intensity), 
														ccnf_expert_intensity(other.ccnf_expert_intensity), cen_expert_intensity(other.cen_expert_intensity),
														early_term_weights(other.early_term_weights), early_term_biases(other.early_term_biases), early_term_cutoffs(other.early_term_cutoffs),
														mirror_inds(other.mirror_inds),mirror_views(other.mirror_views)
{

	// Make sure the matrices are allocated properly
	this->sigma_components.resize(other.sigma_components.size());
	for (size_t i = 0; i < other.sigma_components.size(); ++i)
	{
		this->sigma_components[i].resize(other.sigma_components[i].size());

		for (size_t j = 0; j < other.sigma_components[i].size(); ++j)
		{
			// Make sure the matrix is copied.
			this->sigma_components[i][j] = other.sigma_components[i][j].clone();
		}
	}

	// Make sure the matrices are allocated properly
	this->visibilities.resize(other.visibilities.size());
	for (size_t i = 0; i < other.visibilities.size(); ++i)
	{
		this->visibilities[i].resize(other.visibilities[i].size());

		for (size_t j = 0; j < other.visibilities[i].size(); ++j)
		{
			// Make sure the matrix is copied.
			this->visibilities[i][j] = other.visibilities[i][j].clone();
		}
	}

	preallocated_im2col.resize(other.preallocated_im2col.size());
}

// Returns indices to landmarks that need to have patch responses computed (omits mirrored frontal landmarks for CEN as they will be computed together with their mirrored pair)
std::vector<int> Patch_experts::Collect_visible_landmarks(std::vector<std::vector<cv::Mat_<int> > > visibilities, int scale, int view_id, int n)
{
	std::vector<int> vis_lmk;
	for (int i = 0; i < n; i++)
	{
		if (visibilities[scale][view_id].rows == n)
		{
			if (visibilities[scale][view_id].at<int>(i, 0) != 0)
			{
				// For CEN patch experts and frontal views skip the mirror indices
				if (!cen_expert_intensity.empty())
				{

					// If frontal view we can do mirrored landmarks together
					if (view_id == 0)
					{
						// If the patch expert does not have values, means it's a mirrored version and will be done in another part of a loop
						if (!cen_expert_intensity[scale][view_id][i].biases.empty())
						{
							vis_lmk.push_back(i);
						}
					}
					else
					{
						vis_lmk.push_back(i);
					}
				}
				else
				{
					vis_lmk.push_back(i);
				}
			}
		}
	}
	return vis_lmk;

}

// Returns the patch expert responses given a grayscale image.
// Additionally returns the transform from the image coordinates to the response coordinates (and vice versa).
// The computation also requires the current landmark locations to compute response around, the PDM corresponding to the desired model, and the parameters describing its instance
// Also need to provide the size of the area of interest and the desired scale of analysis
void Patch_experts::Response(std::vector<cv::Mat_<float> >& patch_expert_responses, cv::Matx22f& sim_ref_to_img, 
	cv::Matx22f& sim_img_to_ref, const cv::Mat_<float>& grayscale_image, const PDM& pdm, const cv::Vec6f& params_global,
	const cv::Mat_<float>& params_local, int window_size, int scale)
{

	int view_id = GetViewIdx(params_global, scale);

	int n = pdm.NumberOfPoints();

	// Compute the current landmark locations (around which responses will be computed)
	cv::Mat_<float> landmark_locations;

	pdm.CalcShape2D(landmark_locations, params_local, params_global);

	cv::Mat_<float> reference_shape;

	// Initialise the reference shape on which we'll be warping
	cv::Vec6f global_ref(patch_scaling[scale], 0, 0, 0, 0, 0);

	// Compute the reference shape
	pdm.CalcShape2D(reference_shape, params_local, global_ref);

	// similarity and inverse similarity transform to and from image and reference shape
	cv::Mat_<float> reference_shape_2D = (reference_shape.reshape(1, 2).t());
	cv::Mat_<float> image_shape_2D = landmark_locations.reshape(1, 2).t();

	sim_img_to_ref = Utilities::AlignShapesWithScale(image_shape_2D, reference_shape_2D);
	sim_ref_to_img = sim_img_to_ref.inv(cv::DECOMP_LU);
	
	float a1 = sim_ref_to_img(0, 0);
	float b1 = -sim_ref_to_img(0, 1);

	bool use_ccnf = !this->ccnf_expert_intensity.empty();
	bool use_cen = !this->cen_expert_intensity.empty();

	// If using CCNF patch experts might need to precalculate Sigmas
	if (use_ccnf)
	{
		std::vector<cv::Mat_<float> > sigma_components;

		// Retrieve the correct sigma component size
		for (size_t w_size = 0; w_size < this->sigma_components.size(); ++w_size)
		{
			if (!this->sigma_components[w_size].empty())
			{
				if (window_size*window_size == this->sigma_components[w_size][0].rows)
				{
					sigma_components = this->sigma_components[w_size];
				}
			}
		}

		// Go through all of the landmarks and compute the Sigma for each
		for (int lmark = 0; lmark < n; lmark++)
		{
			// Only for visible landmarks
			if (visibilities[scale][view_id].at<int>(lmark, 0))
			{
				// Precompute sigmas if they are not computed yet
				ccnf_expert_intensity[scale][view_id][lmark].ComputeSigmas(sigma_components, window_size);
			}
		}

	}

	// If using CEN precalculate interpolation matrix
	cv::Mat_<float> interp_mat;
	if (use_cen)
	{
		// Assuming the same size for all experts
		int support_region = 11;
		int area_of_interest_width = window_size + support_region - 1;
		int area_of_interest_height = window_size + support_region - 1;
		int resp_size = area_of_interest_height - support_region + 1;
		interpolationMatrix(interp_mat, resp_size, resp_size, area_of_interest_width, area_of_interest_height);
	}

	// We do not want to create threads for invisible landmarks, so construct an index of visible ones
	std::vector<int> vis_lmk = Collect_visible_landmarks(visibilities, scale, view_id, n);

	// calculate the patch responses for every landmark (this is the heavy lifting of landmark detection)
	parallel_for_(cv::Range(0, vis_lmk.size()), [&](const cv::Range& range) {
		for (int i = range.start; i < range.end; i++)
		{

			// Work out how big the area of interest has to be to get a response of window size
			int area_of_interest_width;
			int area_of_interest_height;
			int ind = vis_lmk.at(i);

			if (use_cen)
			{
				area_of_interest_width = window_size + cen_expert_intensity[scale][view_id][ind].width_support - 1;
				area_of_interest_height = window_size + cen_expert_intensity[scale][view_id][ind].height_support - 1;
			}
			else if (use_ccnf)
			{
				area_of_interest_width = window_size + ccnf_expert_intensity[scale][view_id][ind].width - 1;
				area_of_interest_height = window_size + ccnf_expert_intensity[scale][view_id][ind].height - 1;
			}
			else
			{
				area_of_interest_width = window_size + svr_expert_intensity[scale][view_id][ind].width - 1;
				area_of_interest_height = window_size + svr_expert_intensity[scale][view_id][ind].height - 1;
			}

			// scale and rotate to mean shape to reference frame
			cv::Mat sim = (cv::Mat_<float>(2, 3) << a1, -b1, landmark_locations.at<float>(ind, 0) - a1 * (area_of_interest_width - 1.0f) / 2.0f + b1 * (area_of_interest_width - 1.0f) / 2.0f, b1, a1, landmark_locations.at<float>(ind + n, 0) - a1 * (area_of_interest_width - 1.0f) / 2.0f - b1 * (area_of_interest_width - 1.0f) / 2.0f);

			// Extract the region of interest around the current landmark location
			cv::Mat_<float> area_of_interest(area_of_interest_height, area_of_interest_width, 0.0f);

			cv::warpAffine(grayscale_image, area_of_interest, sim, area_of_interest.size(), cv::WARP_INVERSE_MAP + cv::INTER_LINEAR);

			// Get intensity response either from the SVR, CCNF, or CEN patch experts (prefer CEN as they are the most accurate so far)
			if (!cen_expert_intensity.empty())
			{

				int im2col_size = (area_of_interest_width * area_of_interest_height - 1) / 2;

				cv::Mat_<float> prealloc_mat = preallocated_im2col[ind][im2col_size];

				// If frontal view we can do mirrored landmarks together
				if (view_id == 0)
				{
					// If the patch expert does not have values, means it's a mirrored version and will be done in another part of a loop
					if (!cen_expert_intensity[scale][view_id][ind].biases.empty())
					{
						// No mirrored expert, so do normally
						int mirror_id = mirror_inds.at<int>(ind);
						if (mirror_id == ind)
						{
							cv::Mat_<float> empty(0, 0, 0.0f);
							cen_expert_intensity[scale][view_id][ind].ResponseSparse(area_of_interest, empty, patch_expert_responses[ind], empty, interp_mat, prealloc_mat, empty);
						}
						else
						{

							// Grab mirrored area of interest

							// scale and rotate to mean shape to reference frame
							cv::Mat sim_r = (cv::Mat_<float>(2, 3) << a1, -b1, landmark_locations.at<float>(mirror_id, 0) - a1 * (area_of_interest_width - 1.0f) / 2.0f + b1 * (area_of_interest_width - 1.0f) / 2.0f, b1, a1, landmark_locations.at<float>(mirror_id + n, 0) - a1 * (area_of_interest_width - 1.0f) / 2.0f - b1 * (area_of_interest_width - 1.0f) / 2.0f);

							// Extract the region of interest around the current landmark location
							cv::Mat_<float> area_of_interest_r(area_of_interest_height, area_of_interest_width, 0.0f);

							cv::warpAffine(grayscale_image, area_of_interest_r, sim_r, area_of_interest_r.size(), cv::WARP_INVERSE_MAP + cv::INTER_LINEAR);

							cv::Mat_<float> prealloc_mat_right = preallocated_im2col[mirror_id][im2col_size];

							cen_expert_intensity[scale][view_id][ind].ResponseSparse(area_of_interest, area_of_interest_r, patch_expert_responses[ind], patch_expert_responses[mirror_id], interp_mat, prealloc_mat, prealloc_mat_right);

							preallocated_im2col[mirror_id][im2col_size] = prealloc_mat_right;

						}
					}
				}
				else
				{
					// For space and memory saving use a mirrored patch expert
					if (!cen_expert_intensity[scale][view_id][ind].biases.empty())
					{
						cv::Mat_<float> empty(0, 0, 0.0f);
						cen_expert_intensity[scale][view_id][ind].ResponseSparse(area_of_interest, empty, patch_expert_responses[ind], empty, interp_mat, prealloc_mat, empty);

						// A slower, but slightly more accurate version
						//cen_expert_intensity[scale][view_id][ind].Response(area_of_interest, patch_expert_responses[ind]);
					}
					else
					{
						cv::Mat_<float> empty(0, 0, 0.0f);
						cen_expert_intensity[scale][mirror_views.at<int>(view_id)][mirror_inds.at<int>(ind)].ResponseSparse(empty, area_of_interest, empty, patch_expert_responses[ind], interp_mat, empty, prealloc_mat);
					}
				}

				preallocated_im2col[ind][im2col_size] = prealloc_mat;

			}
			else if (!ccnf_expert_intensity.empty())
			{
				// get the correct size response window			
				patch_expert_responses[ind] = cv::Mat_<float>(window_size, window_size);

				int im2col_size = area_of_interest_width * area_of_interest_height;

				cv::Mat_<float> prealloc_mat = preallocated_im2col[ind][im2col_size];

				ccnf_expert_intensity[scale][view_id][ind].ResponseOpenBlas(area_of_interest, patch_expert_responses[ind], prealloc_mat);

				preallocated_im2col[ind][im2col_size] = prealloc_mat;

				// Below is an alternative way to compute the same, but that uses FFT instead of OpenBLAS
				// ccnf_expert_intensity[scale][view_id][ind].Response(area_of_interest, patch_expert_responses[ind]);

			}
			else
			{
				// get the correct size response window			
				patch_expert_responses[ind] = cv::Mat_<float>(window_size, window_size);

				svr_expert_intensity[scale][view_id][ind].Response(area_of_interest, patch_expert_responses[ind]);
			}
		}
	});
}


//=============================================================================
// Getting the closest view center based on orientation
int Patch_experts::GetViewIdx(const cv::Vec6f& params_global, int scale) const
{	
	int idx = 0;
	
	float dbest;

	for(int i = 0; i < this->nViews(scale); i++)
	{
		float v1 = params_global[1] - centers[scale][i][0]; 
		float v2 = params_global[2] - centers[scale][i][1];
		float v3 = params_global[3] - centers[scale][i][2];
			
		float d = v1*v1 + v2*v2 + v3*v3;

		if(i == 0 || d < dbest)
		{
			dbest = d;
			idx = i;
		}
	}
	return idx;
}


//===========================================================================
bool Patch_experts::Read(std::vector<std::string> intensity_svr_expert_locations, std::vector<std::string> intensity_ccnf_expert_locations,
	std::vector<std::string> intensity_cen_expert_locations, std::string early_term_loc)
{

	// initialise the SVR intensity patch expert parameters
	int num_intensity_svr = intensity_svr_expert_locations.size();
	centers.resize(num_intensity_svr);
	visibilities.resize(num_intensity_svr);
	patch_scaling.resize(num_intensity_svr);
	
	svr_expert_intensity.resize(num_intensity_svr);
	
	// Reading in SVR intensity patch experts for each scales it is defined in
	for(int scale = 0; scale < num_intensity_svr; ++scale)
	{		
		std::string location = intensity_svr_expert_locations[scale];
		//std::cout << "Reading the intensity SVR patch experts from: " << location << "....";
		bool success_read = Read_SVR_patch_experts(location,  centers[scale], visibilities[scale], svr_expert_intensity[scale], patch_scaling[scale]);
		if (!success_read)
		{
			return false;
		}
	}

	// Initialise and read CCNF patch experts (currently only intensity based), 
	int num_intensity_ccnf = intensity_ccnf_expert_locations.size();

	// CCNF experts override the SVR ones
	if(num_intensity_ccnf > 0)
	{
		centers.resize(num_intensity_ccnf);
		visibilities.resize(num_intensity_ccnf);
		patch_scaling.resize(num_intensity_ccnf);
		ccnf_expert_intensity.resize(num_intensity_ccnf);
	}

	for(int scale = 0; scale < num_intensity_ccnf; ++scale)
	{		
		std::string location = intensity_ccnf_expert_locations[scale];
		//std::cout << "Reading the intensity CCNF patch experts from: " << location << "....";
		bool success_read = Read_CCNF_patch_experts(location,  centers[scale], visibilities[scale], ccnf_expert_intensity[scale], patch_scaling[scale]);

		if (!success_read)
		{
			return false;
		}

		if (scale == 0)
		{
			preallocated_im2col.resize(ccnf_expert_intensity[0][0].size());
		}
	}

	// Initialise and read CEN patch experts (currently only intensity based), 
	int num_intensity_cen = intensity_cen_expert_locations.size();

	// CEN experts override the SVR and CCNF ones
	if (num_intensity_cen > 0)
	{
		centers.resize(num_intensity_cen);
		visibilities.resize(num_intensity_cen);
		patch_scaling.resize(num_intensity_cen);
		cen_expert_intensity.resize(num_intensity_cen);
	}

	for (int scale = 0; scale < num_intensity_cen; ++scale)
	{
		std::string location = intensity_cen_expert_locations[scale];
		//std::cout << "Reading the intensity CEN patch experts from: " << location << "....";
		bool success_read = Read_CEN_patch_experts(location, centers[scale], visibilities[scale], cen_expert_intensity[scale], patch_scaling[scale]);
		if (!success_read)
		{
			return false;
		}

		if (scale == 0)
		{
			preallocated_im2col.resize(cen_expert_intensity[0][0].size());
		}

	}


	// Reading in early termination parameters
	if (!early_term_loc.empty())
	{
		std::ifstream earlyTermFile(early_term_loc.c_str(), std::ios_base::in);

		if (!earlyTermFile.is_open())
		{
			return false;
		}

		// Reading in weights/biases/cutoffs
		for (size_t i = 0; i < centers[0].size(); ++i)
		{
			double weight;
			earlyTermFile >> weight;
			early_term_weights.push_back(weight);
		}

		for (size_t i = 0; i < centers[0].size(); ++i)
		{
			double bias;
			earlyTermFile >> bias;
			early_term_biases.push_back(bias);
		}

		for (size_t i = 0; i < centers[0].size(); ++i)
		{
			double cutoff;
			earlyTermFile >> cutoff;
			early_term_cutoffs.push_back(cutoff);
		}
	}
	return true;
}
//======================= Reading the SVR patch experts =========================================//
bool Patch_experts::Read_SVR_patch_experts(std::string expert_location, std::vector<cv::Vec3d>& centers,
	std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<Multi_SVR_patch_expert> >& patches, double& scale)
{

	std::ifstream patchesFile(expert_location.c_str(), std::ios_base::in);

	if(patchesFile.is_open())
	{
		LandmarkDetector::SkipComments(patchesFile);

		patchesFile >> scale;

		LandmarkDetector::SkipComments(patchesFile);

		int numberViews;		

		patchesFile >> numberViews; 

		// read the visibility
		centers.resize(numberViews);
		visibility.resize(numberViews);
  
		patches.resize(numberViews);

		LandmarkDetector::SkipComments(patchesFile);

		// centers of each view (which view corresponds to which orientation)
		for(size_t i = 0; i < centers.size(); i++)
		{
			cv::Mat center;
			LandmarkDetector::ReadMat(patchesFile, center);	
			center.copyTo(centers[i]);
			centers[i] = centers[i] * M_PI / 180.0;
		}

		LandmarkDetector::SkipComments(patchesFile);

		// the visibility of points for each of the views (which verts are visible at a specific view
		for(size_t i = 0; i < visibility.size(); i++)
		{
			LandmarkDetector::ReadMat(patchesFile, visibility[i]);				
		}

		int numberOfPoints = visibility[0].rows;

		LandmarkDetector::SkipComments(patchesFile);

		// read the patches themselves
		for(size_t i = 0; i < patches.size(); i++)
		{
			// number of patches for each view
			patches[i].resize(numberOfPoints);
			// read in each patch
			for(int j = 0; j < numberOfPoints; j++)
			{
				patches[i][j].Read(patchesFile);
			}
		}
	
		//std::cout << "Done" << std::endl;
		return true;
	}
	else
	{
		std::cout << "Can't find/open the patches file" << std::endl;
		return false;
	}
}

//======================= Reading the CCNF patch experts =========================================//
bool Patch_experts::Read_CCNF_patch_experts(std::string patchesFileLocation, std::vector<cv::Vec3d>& centers,
	std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<CCNF_patch_expert> >& patches, double& patchScaling)
{

	std::ifstream patchesFile(patchesFileLocation.c_str(), std::ios::in | std::ios::binary);

	if(patchesFile.is_open())
	{
		patchesFile.read ((char*)&patchScaling, 8);
		
		int numberViews;		
		patchesFile.read ((char*)&numberViews, 4);

		// read the visibility
		centers.resize(numberViews);
		visibility.resize(numberViews);
  
		patches.resize(numberViews);
		
		// centers of each view (which view corresponds to which orientation)
		for(size_t i = 0; i < centers.size(); i++)
		{
			cv::Mat center;
			LandmarkDetector::ReadMatBin(patchesFile, center);	
			center.copyTo(centers[i]);
			centers[i] = centers[i] * M_PI / 180.0;
		}

		// the visibility of points for each of the views (which verts are visible at a specific view
		for(size_t i = 0; i < visibility.size(); i++)
		{
			LandmarkDetector::ReadMatBin(patchesFile, visibility[i]);				
		}
		int numberOfPoints = visibility[0].rows;

		// Read the possible SigmaInvs (without beta), this will be followed by patch reading (this assumes all of them have the same type, and number of betas)
		int num_win_sizes;
		int num_sigma_comp;
		patchesFile.read ((char*)&num_win_sizes, 4);

		std::vector<int> windows;
		windows.resize(num_win_sizes);

		std::vector<std::vector<cv::Mat_<float> > > sigma_components;
		sigma_components.resize(num_win_sizes);

		for (int w=0; w < num_win_sizes; ++w)
		{
			patchesFile.read ((char*)&windows[w], 4);

			patchesFile.read ((char*)&num_sigma_comp, 4);

			sigma_components[w].resize(num_sigma_comp);

			for(int s=0; s < num_sigma_comp; ++s)
			{
				LandmarkDetector::ReadMatBin(patchesFile, sigma_components[w][s]);
			}
		}
		
		this->sigma_components = sigma_components;

		// read the patches themselves
		for(size_t i = 0; i < patches.size(); i++)
		{
			// number of patches for each view
			patches[i].resize(numberOfPoints);
			// read in each patch
			for(int j = 0; j < numberOfPoints; j++)
			{
				patches[i][j].Read(patchesFile, windows, sigma_components);
			}
		}
		//std::cout << "Done" << std::endl;
		return true;
	}
	else
	{
		std::cout << "Can't find/open the patches file" << std::endl;
		return false;
	}
}

//======================= Reading the CEN patch experts =========================================//
bool Patch_experts::Read_CEN_patch_experts(std::string expert_location, std::vector<cv::Vec3d>& centers, 
	std::vector<cv::Mat_<int> >& visibility, std::vector<std::vector<CEN_patch_expert> >& patches, double& scale)
{

	std::ifstream patchesFile(expert_location.c_str(), std::ios::in | std::ios::binary);

	if (patchesFile.is_open())
	{
		patchesFile.read((char*)&scale, 8);

		int numberViews;
		patchesFile.read((char*)&numberViews, 4);

		// read the visibility
		centers.resize(numberViews);
		visibility.resize(numberViews);

		patches.resize(numberViews);

		// centers of each view (which view corresponds to which orientation)
		for (size_t i = 0; i < centers.size(); i++)
		{
			cv::Mat center;
			LandmarkDetector::ReadMatBin(patchesFile, center);
			center.copyTo(centers[i]);
			centers[i] = centers[i] * M_PI / 180.0;
		}

		// the visibility of points for each of the views (which verts are visible at a specific view
		for (size_t i = 0; i < visibility.size(); i++)
		{
			LandmarkDetector::ReadMatBin(patchesFile, visibility[i]);
		}
		int numberOfPoints = visibility[0].rows;
		
		LandmarkDetector::ReadMatBin(patchesFile, mirror_inds);
		LandmarkDetector::ReadMatBin(patchesFile, mirror_views);

		// read the patches themselves
		for (size_t i = 0; i < patches.size(); i++)
		{
			// number of patches for each view
			patches[i].resize(numberOfPoints);
			// read in each patch
			for (int j = 0; j < numberOfPoints; j++)
			{
				patches[i][j].Read(patchesFile);
			}
		}
		//std::cout << "Done" << std::endl;
		return true;
	}
	else
	{
		std::cout << "Could not find CEN patch experts, for instructions of how to download them, see https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download \n" << std::endl;
		return false;
	}
}
