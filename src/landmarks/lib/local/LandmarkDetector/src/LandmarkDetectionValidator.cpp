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

#include "LandmarkDetectionValidator.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// System includes
#include <fstream>

// Math includes
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
// Local includes
#include "LandmarkDetectorUtils.h"
#include "CNN_utils.h"

using namespace LandmarkDetector;

// Copy constructor
DetectionValidator::DetectionValidator(const DetectionValidator& other) : orientations(other.orientations), paws(other.paws),
cnn_subsampling_layers(other.cnn_subsampling_layers), cnn_layer_types(other.cnn_layer_types), cnn_convolutional_layers_im2col_precomp(other.cnn_convolutional_layers_im2col_precomp),
cnn_convolutional_layers_weights(other.cnn_convolutional_layers_weights)
{

	this->cnn_convolutional_layers.resize(other.cnn_convolutional_layers.size());
	for (size_t v = 0; v < other.cnn_convolutional_layers.size(); ++v)
	{
		this->cnn_convolutional_layers[v].resize(other.cnn_convolutional_layers[v].size());

		for (size_t l = 0; l < other.cnn_convolutional_layers[v].size(); ++l)
		{
			this->cnn_convolutional_layers[v][l].resize(other.cnn_convolutional_layers[v][l].size());

			for (size_t i = 0; i < other.cnn_convolutional_layers[v][l].size(); ++i)
			{
				this->cnn_convolutional_layers[v][l][i].resize(other.cnn_convolutional_layers[v][l][i].size());

				for (size_t k = 0; k < other.cnn_convolutional_layers[v][l][i].size(); ++k)
				{
					// Make sure the matrix is copied.
					this->cnn_convolutional_layers[v][l][i][k] = other.cnn_convolutional_layers[v][l][i][k].clone();
				}

			}
		}
	}

	this->cnn_fully_connected_layers_weights.resize(other.cnn_fully_connected_layers_weights.size());
	for (size_t v = 0; v < other.cnn_fully_connected_layers_weights.size(); ++v)
	{
		this->cnn_fully_connected_layers_weights[v].resize(other.cnn_fully_connected_layers_weights[v].size());

		for (size_t l = 0; l < other.cnn_fully_connected_layers_weights[v].size(); ++l)
		{
			// Make sure the matrix is copied.
			this->cnn_fully_connected_layers_weights[v][l] = other.cnn_fully_connected_layers_weights[v][l].clone();
		}
	}

	this->cnn_fully_connected_layers_biases.resize(other.cnn_fully_connected_layers_biases.size());
	for (size_t v = 0; v < other.cnn_fully_connected_layers_biases.size(); ++v)
	{
		this->cnn_fully_connected_layers_biases[v].resize(other.cnn_fully_connected_layers_biases[v].size());

		for (size_t l = 0; l < other.cnn_fully_connected_layers_biases[v].size(); ++l)
		{
			// Make sure the matrix is copied.
			this->cnn_fully_connected_layers_biases[v][l] = other.cnn_fully_connected_layers_biases[v][l].clone();
		}
	}

	this->mean_images.resize(other.mean_images.size());
	for (size_t i = 0; i < other.mean_images.size(); ++i)
	{
		// Make sure the matrix is copied.
		this->mean_images[i] = other.mean_images[i].clone();
	}

	this->standard_deviations.resize(other.standard_deviations.size());
	for (size_t i = 0; i < other.standard_deviations.size(); ++i)
	{
		// Make sure the matrix is copied.
		this->standard_deviations[i] = other.standard_deviations[i].clone();
	}

}

//===========================================================================
// Read in the landmark detection validation module
void DetectionValidator::Read(std::string location)
{

	std::ifstream detection_validator_stream (location, std::ios::in | std::ios::binary);
	if (detection_validator_stream.is_open())	
	{				
		detection_validator_stream.seekg (0, std::ios::beg);

		// Read validator type
		int validator_type;
		detection_validator_stream.read ((char*)&validator_type, 4);

		if (validator_type != 3)
		{
			std::cout << "ERROR: Using old face validator, no longer supported" << std::endl;
		}

		// Read the number of views (orientations) within the validator
		int n;
		detection_validator_stream.read ((char*)&n, 4);
	
		orientations.resize(n);

		for(int i = 0; i < n; i++)
		{
			cv::Mat_<double> orientation_tmp;
			LandmarkDetector::ReadMatBin(detection_validator_stream, orientation_tmp);		
		
			orientations[i] = cv::Vec3d(orientation_tmp.at<double>(0), orientation_tmp.at<double>(1), orientation_tmp.at<double>(2));

			// Convert from degrees to radians
			orientations[i] = orientations[i] * M_PI / 180.0;
		}

		// Initialise the piece-wise affine warps, biases and weights
		paws.resize(n);

		cnn_convolutional_layers_weights.resize(n);
		cnn_convolutional_layers_im2col_precomp.resize(n);
		cnn_convolutional_layers.resize(n);
		cnn_fully_connected_layers_weights.resize(n);
		cnn_layer_types.resize(n);
		cnn_fully_connected_layers_biases.resize(n);

		// Initialise the normalisation terms
		mean_images.resize(n);
		standard_deviations.resize(n);

		// Read in the validators for each of the views
		for(int i = 0; i < n; i++)
		{

			// Read in the mean images
			cv::Mat_<double> mean_img;
			LandmarkDetector::ReadMatBin(detection_validator_stream, mean_img);
			mean_img.convertTo(mean_images[i], CV_32F);
			mean_images[i] = mean_images[i].t();

			cv::Mat_<double> std_dev;
			LandmarkDetector::ReadMatBin(detection_validator_stream, std_dev);
			std_dev.convertTo(standard_deviations[i], CV_32F);

			standard_deviations[i] = standard_deviations[i].t();

			// Model specifics
			if (validator_type == 3)
			{
				int network_depth;
				detection_validator_stream.read((char*)&network_depth, 4);

				cnn_layer_types[i].resize(network_depth);

				for (int layer = 0; layer < network_depth; ++layer)
				{

					int layer_type;
					detection_validator_stream.read((char*)&layer_type, 4);
					cnn_layer_types[i][layer] = layer_type;

					// convolutional
					if (layer_type == 0)
					{

						// Read the number of input maps
						int num_in_maps;
						detection_validator_stream.read((char*)&num_in_maps, 4);

						// Read the number of kernels for each input map
						int num_kernels;
						detection_validator_stream.read((char*)&num_kernels, 4);

						std::vector<std::vector<cv::Mat_<float> > > kernels;

						kernels.resize(num_in_maps);

						std::vector<float> biases;
						for (int k = 0; k < num_kernels; ++k)
						{
							float bias;
							detection_validator_stream.read((char*)&bias, 4);
							biases.push_back(bias);
						}

						// For every input map
						for (int in = 0; in < num_in_maps; ++in)
						{
							kernels[in].resize(num_kernels);

							// For every kernel on that input map
							for (int k = 0; k < num_kernels; ++k)
							{
								ReadMatBin(detection_validator_stream, kernels[in][k]);

							}
						}

						cnn_convolutional_layers[i].push_back(kernels);

						// Rearrange the kernels for faster inference with FFT
						std::vector<std::vector<cv::Mat_<float> > > kernels_rearr;
						kernels_rearr.resize(num_kernels);

						// Fill up the rearranged layer
						for (int k = 0; k < num_kernels; ++k)
						{
							for (int in = 0; in < num_in_maps; ++in)
							{
								kernels_rearr[k].push_back(kernels[in][k]);
							}
						}

						// Rearrange the flattened kernels into weight matrices for direct convolution computation
						cv::Mat_<float> weight_matrix(num_in_maps * kernels_rearr[0][0].rows * kernels_rearr[0][0].cols, num_kernels);
						for (int k = 0; k < num_kernels; ++k)
						{
							for (int i = 0; i < num_in_maps; ++i)
							{
								// Flatten the kernel
								cv::Mat_<float> k_flat = kernels_rearr[k][i].t();
								k_flat = k_flat.reshape(0, 1).t();
								k_flat.copyTo(weight_matrix(cv::Rect(k, i * kernels_rearr[0][0].rows * kernels_rearr[0][0].cols, 1, kernels_rearr[0][0].rows * kernels_rearr[0][0].cols)));
							}
						}

						// Transpose the weight matrix for more convenient computation
						weight_matrix = weight_matrix.t();

						// Add a bias term to the weight matrix for efficiency
						cv::Mat_<float> W(weight_matrix.rows, weight_matrix.cols + 1, 1.0);
						for (int k = 0; k < weight_matrix.rows; ++k)
						{
							W.at<float>(k, weight_matrix.cols) = biases[k];
						}
						weight_matrix.copyTo(W(cv::Rect(0, 0, weight_matrix.cols, weight_matrix.rows)));

						cnn_convolutional_layers_weights[i].push_back(W.t());
						cnn_convolutional_layers_im2col_precomp[i].push_back(cv::Mat_<float>());
					}
					else if (layer_type == 2)
					{
						cv::Mat_<float> biases;
						ReadMatBin(detection_validator_stream, biases);
						cnn_fully_connected_layers_biases[i].push_back(biases);

						// Fully connected layer
						cv::Mat_<float> weights;
						ReadMatBin(detection_validator_stream, weights);
						cnn_fully_connected_layers_weights[i].push_back(weights);
					}
				}
			}
			// Read in the piece-wise affine warps
			paws[i].Read(detection_validator_stream);
		}
		
	}
	else
	{
		std::cout << "WARNING: Can't find the Face checker location" << std::endl;
	}
}

//===========================================================================
// Check if the fitting actually succeeded
float DetectionValidator::Check(const cv::Vec3d& orientation, const cv::Mat_<uchar>& intensity_img, cv::Mat_<float>& detected_landmarks)
{

	int id = GetViewId(orientation);
	
	// The warped (cropped) image, corresponding to a face lying withing the detected lanmarks
	cv::Mat_<float> warped;
	
	// First only use the ROI of the image of interest
	cv::Mat_<float> detected_landmarks_local = detected_landmarks.clone();

	float min_x_f, max_x_f, min_y_f, max_y_f;
	ExtractBoundingBox(detected_landmarks_local, min_x_f, max_x_f, min_y_f, max_y_f);

	cv::Mat_<float> xs = detected_landmarks_local(cv::Rect(0, 0, 1, detected_landmarks.rows / 2));
	cv::Mat_<float> ys = detected_landmarks_local(cv::Rect(0, detected_landmarks.rows / 2, 1, detected_landmarks.rows / 2));

	// Picking the ROI (some extra space for bilinear interpolation)
	int min_x = (int)(min_x_f - 3.0f);
	int max_x = (int)(max_x_f + 3.0f);
	int min_y = (int)(min_y_f - 3.0f);
	int max_y = (int)(max_y_f + 3.0f);

	if (min_x < 0) min_x = 0;
	if (min_y < 0) min_y = 0;
	if (max_x > intensity_img.cols - 1) max_x = intensity_img.cols - 1;
	if (max_y > intensity_img.rows - 1) max_y = intensity_img.rows - 1;
	xs = xs - min_x;
	ys = ys - min_y;

	// If the ROI is non existent return failure (this could happen if all landmarks are outside of the image)
	if (max_x - min_x <= 1 || max_y - min_y <= 1)
	{
		return 0.0f;
	}

	cv::Mat_<float> intensity_img_float_local;
	intensity_img(cv::Rect(min_x, min_y, max_x - min_x, max_y - min_y)).convertTo(intensity_img_float_local, CV_32F);

	// the piece-wise affine image warping
	paws[id].Warp(intensity_img_float_local, warped, detected_landmarks_local);

	// The actual validation step
	double dec = CheckCNN(warped, id);

	// Convert it to a more interpretable signal (0 low confidence, 1 high confidence)
	dec = 0.5 * (1.0 - dec);

	return (float)dec;
}

double DetectionValidator::CheckCNN(const cv::Mat_<float>& warped_img, int view_id)
{

	cv::Mat_<float> feature_vec;
	NormaliseWarpedToVector(warped_img, feature_vec, view_id);

	// Create a normalised image from the crop vector
	cv::Mat_<float> img(warped_img.size(), 0.0);
	img = img.t();

	cv::Mat mask = paws[view_id].pixel_mask.t();
	cv::MatIterator_<uchar>  mask_it = mask.begin<uchar>();

	cv::MatIterator_<float> feature_it = feature_vec.begin();
	cv::MatIterator_<float> img_it = img.begin();

	int wInt = img.cols;
	int hInt = img.rows;

	for (int i = 0; i < wInt; ++i)
	{
		for (int j = 0; j < hInt; ++j, ++mask_it, ++img_it)
		{
			// if is within mask
			if (*mask_it)
			{
				// assign the feature to image if it is within the mask
				*img_it = (float)*feature_it++;
			}
		}
	}
	img = img.t();

	int cnn_layer = 0;
	int fully_connected_layer = 0;

	std::vector<cv::Mat_<float> > input_maps;
	input_maps.push_back(img);

	std::vector<cv::Mat_<float> > outputs;

	for (size_t layer = 0; layer < cnn_layer_types[view_id].size(); ++layer)
	{
		// Determine layer type
		int layer_type = cnn_layer_types[view_id][layer];

		// Convolutional layer
		if (layer_type == 0)
		{

			convolution_direct_blas(outputs, input_maps, cnn_convolutional_layers_weights[view_id][cnn_layer], cnn_convolutional_layers[view_id][cnn_layer][0][0].rows, cnn_convolutional_layers[view_id][cnn_layer][0][0].cols, cnn_convolutional_layers_im2col_precomp[view_id][cnn_layer]);

			cnn_layer++;
		}
		if (layer_type == 1)
		{
			max_pooling(outputs, input_maps, 2, 2, 2, 2);
		}
		if (layer_type == 2)
		{

			fully_connected(outputs, input_maps, cnn_fully_connected_layers_weights[view_id][fully_connected_layer].t(), cnn_fully_connected_layers_biases[view_id][fully_connected_layer]);
			fully_connected_layer++;
		}
		if (layer_type == 3) // ReLU
		{
			outputs.clear();
			for (size_t k = 0; k < input_maps.size(); ++k)
			{
				// Apply the ReLU
				cv::threshold(input_maps[k], input_maps[k], 0, 0, cv::THRESH_TOZERO);
				outputs.push_back(input_maps[k]);

			}
		}
		if (layer_type == 4)
		{
			outputs.clear();
			for (size_t k = 0; k < input_maps.size(); ++k)
			{
				// Apply the sigmoid
				cv::exp(-input_maps[k], input_maps[k]);
				input_maps[k] = 1.0 / (1.0 + input_maps[k]);

				outputs.push_back(input_maps[k]);

			}
		}
		// Set the outputs of this layer to inputs of the next
		input_maps = outputs;

	}

	// Convert the class label to a continuous value
	double max_val = 0;
	cv::Point max_loc;
	cv::minMaxLoc(outputs[0].t(), 0, &max_val, 0, &max_loc);
	int max_idx = max_loc.y;
	double max = 1;
	double min = -1;
	double bins = (double)outputs[0].cols;
	// Unquantizing the softmax layer to continuous value
	double step_size = (max - min) / bins; // This should be saved somewhere
	double unquantized = min + step_size / 2.0 + max_idx * step_size;

	return unquantized;
}

void DetectionValidator::NormaliseWarpedToVector(const cv::Mat_<float>& warped_img, cv::Mat_<float>& feature_vec, int view_id)
{
	cv::Mat_<float> warped_t = warped_img.t();
	
	// the vector to be filled with paw values
	cv::MatIterator_<float> vp;	
	cv::MatIterator_<float>  cp;

	cv::Mat_<float> vec(paws[view_id].number_of_pixels,1);
	vp = vec.begin();

	cp = warped_t.begin();		

	int wInt = warped_img.cols;
	int hInt = warped_img.rows;

	// the mask indicating if point is within or outside the face region
	
	cv::Mat maskT = paws[view_id].pixel_mask.t();

	cv::MatIterator_<uchar>  mp = maskT.begin<uchar>();

	for(int i=0; i < wInt; ++i)
	{
		for(int j=0; j < hInt; ++j, ++mp, ++cp)
		{
			// if is within mask
			if(*mp)
			{
				*vp++ = *cp;
			}
		}
	}

	// Local normalisation
	cv::Scalar mean;
	cv::Scalar std;
	cv::meanStdDev(vec, mean, std);

	// subtract the mean image
	vec -= mean[0];

	// Normalise the image
	if(std[0] == 0)
	{
		std[0] = 1;
	}
	
	vec /= std[0];

	// Global normalisation
	feature_vec = (vec - mean_images[view_id])  / standard_deviations[view_id];
}

// Getting the closest view center based on orientation
int DetectionValidator::GetViewId(const cv::Vec3d& orientation) const
{
	int id = 0;

	double dbest = -1.0;

	for(size_t i = 0; i < this->orientations.size(); i++)
	{
	
		// Distance to current view
		double d = cv::norm(orientation, this->orientations[i]);

		if(i == 0 || d < dbest)
		{
			dbest = d;
			id = i;
		}
	}
	return id;
	
}


