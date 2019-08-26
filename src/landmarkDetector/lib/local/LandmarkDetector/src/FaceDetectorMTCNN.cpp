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

#include "FaceDetectorMTCNN.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#include "LandmarkDetectorUtils.h"

// CNN includes
#include "CNN_utils.h"

// Instead of including cblas.h (the definitions from OpenBLAS and other BLAS libraries differ, declare the required OpenBLAS functionality here)
#ifdef __cplusplus
extern "C" {
	/* Assume C declarations for C++ */
#endif  /* __cplusplus */

	/*Set the number of threads on runtime.*/
	void openblas_set_num_threads(int num_threads);
}

using namespace LandmarkDetector;

// Constructor from model file location
FaceDetectorMTCNN::FaceDetectorMTCNN(const std::string& location)
{
	this->Read(location);
}
// Copy constructor
FaceDetectorMTCNN::FaceDetectorMTCNN(const FaceDetectorMTCNN& other) : PNet(other.PNet), RNet(other.RNet), ONet(other.ONet)
{
}

CNN::CNN(const CNN& other) : cnn_layer_types(other.cnn_layer_types), cnn_max_pooling_layers(other.cnn_max_pooling_layers), cnn_convolutional_layers_bias(other.cnn_convolutional_layers_bias), conv_layer_pre_alloc_im2col(other.conv_layer_pre_alloc_im2col)
{

	this->cnn_convolutional_layers_weights.resize(other.cnn_convolutional_layers_weights.size());
	for (size_t l = 0; l < other.cnn_convolutional_layers_weights.size(); ++l)
	{
		// Make sure the matrix is copied.
		this->cnn_convolutional_layers_weights[l] = other.cnn_convolutional_layers_weights[l].clone();
	}

	this->cnn_convolutional_layers.resize(other.cnn_convolutional_layers.size());
	for (size_t l = 0; l < other.cnn_convolutional_layers.size(); ++l)
	{
		this->cnn_convolutional_layers[l].resize(other.cnn_convolutional_layers[l].size());

		for (size_t i = 0; i < other.cnn_convolutional_layers[l].size(); ++i)
		{
			this->cnn_convolutional_layers[l][i].resize(other.cnn_convolutional_layers[l][i].size());

			for (size_t k = 0; k < other.cnn_convolutional_layers[l][i].size(); ++k)
			{
				// Make sure the matrix is copied.
				this->cnn_convolutional_layers[l][i][k] = other.cnn_convolutional_layers[l][i][k].clone();
			}
		}
	}

	this->cnn_fully_connected_layers_weights.resize(other.cnn_fully_connected_layers_weights.size());

	for (size_t l = 0; l < other.cnn_fully_connected_layers_weights.size(); ++l)
	{
		// Make sure the matrix is copied.
		this->cnn_fully_connected_layers_weights[l] = other.cnn_fully_connected_layers_weights[l].clone();
	}

	this->cnn_fully_connected_layers_biases.resize(other.cnn_fully_connected_layers_biases.size());

	for (size_t l = 0; l < other.cnn_fully_connected_layers_biases.size(); ++l)
	{
		// Make sure the matrix is copied.
		this->cnn_fully_connected_layers_biases[l] = other.cnn_fully_connected_layers_biases[l].clone();
	}

	this->cnn_prelu_layer_weights.resize(other.cnn_prelu_layer_weights.size());

	for (size_t l = 0; l < other.cnn_prelu_layer_weights.size(); ++l)
	{
		// Make sure the matrix is copied.
		this->cnn_prelu_layer_weights[l] = other.cnn_prelu_layer_weights[l].clone();
	}
}

std::vector<cv::Mat_<float>> CNN::Inference(const cv::Mat& input_img, bool direct, bool thread_safe)
{
	if (input_img.channels() == 1)
	{
		cv::cvtColor(input_img, input_img, cv::COLOR_GRAY2BGR);
	}

	int cnn_layer = 0;
	int fully_connected_layer = 0;
	int prelu_layer = 0;
	int max_pool_layer = 0;

	// Slit a BGR image into three chnels
	cv::Mat channels[3]; 
	cv::split(input_img, channels);  

	// Flip the BGR order to RGB
	std::vector<cv::Mat_<float> > input_maps;
	input_maps.push_back(channels[2]);
	input_maps.push_back(channels[1]);
	input_maps.push_back(channels[0]);

	std::vector<cv::Mat_<float> > outputs;

	for (size_t layer = 0; layer < cnn_layer_types.size(); ++layer)
	{

		// Determine layer type
		int layer_type = cnn_layer_types[layer];

		// Convolutional layer
		if (layer_type == 0)		
		{

			// Either perform direct convolution through matrix multiplication or use an FFT optimized version, which one is optimal depends on the kernel and input sizes
			if (direct)
			{
				if(thread_safe)
				{
					cv::Mat_<float> pre_alloc;
					convolution_direct_blas(outputs, input_maps, cnn_convolutional_layers_weights[cnn_layer], cnn_convolutional_layers[cnn_layer][0][0].rows, cnn_convolutional_layers[cnn_layer][0][0].cols, pre_alloc);
				}
				else
				{
					convolution_direct_blas(outputs, input_maps, cnn_convolutional_layers_weights[cnn_layer], cnn_convolutional_layers[cnn_layer][0][0].rows, cnn_convolutional_layers[cnn_layer][0][0].cols, conv_layer_pre_alloc_im2col[cnn_layer]);
				}
	
			}
			else
			{
				convolution_fft2(outputs, input_maps, cnn_convolutional_layers[cnn_layer], cnn_convolutional_layers_bias[cnn_layer], cnn_convolutional_layers_dft[cnn_layer]);
			}
			//vector<cv::Mat_<float> > outs;
			//convolution_fft(outs, input_maps, cnn_convolutional_layers[cnn_layer], cnn_convolutional_layers_bias[cnn_layer], cnn_convolutional_layers_dft[cnn_layer]);



			cnn_layer++;
		}
		if (layer_type == 1)
		{

			int stride_x = std::get<2>(cnn_max_pooling_layers[max_pool_layer]);
			int stride_y = std::get<3>(cnn_max_pooling_layers[max_pool_layer]);
			
			int kernel_size_x = std::get<0>(cnn_max_pooling_layers[max_pool_layer]);
			int kernel_size_y = std::get<1>(cnn_max_pooling_layers[max_pool_layer]);

			max_pooling(outputs, input_maps, stride_x, stride_y, kernel_size_x, kernel_size_y);
			max_pool_layer++;
		}
		if (layer_type == 2)
		{
			fully_connected(outputs, input_maps, cnn_fully_connected_layers_weights[fully_connected_layer], cnn_fully_connected_layers_biases[fully_connected_layer]);
			fully_connected_layer++;
		}
		if (layer_type == 3) // PReLU
		{
			// In place prelu computation
			PReLU(input_maps, cnn_prelu_layer_weights[prelu_layer]);
			outputs = input_maps;
			prelu_layer++;
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
		// Set the outputs of this layer to inputs of the next one
		input_maps = outputs;		
	}

	
	return outputs;

}

void ReadMatBin(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row, col, type;

	stream.read((char*)&row, 4);
	stream.read((char*)&col, 4);
	stream.read((char*)&type, 4);

	output_mat = cv::Mat(row, col, type);
	int size = output_mat.rows * output_mat.cols * output_mat.elemSize();
	stream.read((char *)output_mat.data, size);

}

void CNN::ClearPrecomp()
{
	for (size_t k1 = 0; k1 < cnn_convolutional_layers_dft.size(); ++k1)
	{
		for (size_t k2 = 0; k2 < cnn_convolutional_layers_dft[k1].size(); ++k2)
		{
			cnn_convolutional_layers_dft[k1][k2].clear();
		}
	}
}

void CNN::Read(const std::string& location)
{

	openblas_set_num_threads(1);

	std::ifstream cnn_stream(location, std::ios::in | std::ios::binary);
	if (cnn_stream.is_open())
	{
		cnn_stream.seekg(0, std::ios::beg);

		// Reading in CNNs

		int network_depth;
		cnn_stream.read((char*)&network_depth, 4);

		cnn_layer_types.resize(network_depth);

		for (int layer = 0; layer < network_depth; ++layer)
		{

			int layer_type;
			cnn_stream.read((char*)&layer_type, 4);
			cnn_layer_types[layer] = layer_type;

			// convolutional
			if (layer_type == 0)
			{

				// Read the number of input maps
				int num_in_maps;
				cnn_stream.read((char*)&num_in_maps, 4);

				// Read the number of kernels for each input map
				int num_kernels;
				cnn_stream.read((char*)&num_kernels, 4);

				std::vector<std::vector<cv::Mat_<float> > > kernels;

				kernels.resize(num_in_maps);

				std::vector<float> biases;
				for (int k = 0; k < num_kernels; ++k)
				{
					float bias;
					cnn_stream.read((char*)&bias, 4);
					biases.push_back(bias);
				}

				cnn_convolutional_layers_bias.push_back(biases);

				// For every input map
				for (int in = 0; in < num_in_maps; ++in)
				{
					kernels[in].resize(num_kernels);

					// For every kernel on that input map
					for (int k = 0; k < num_kernels; ++k)
					{
						ReadMatBin(cnn_stream, kernels[in][k]);

					}
				}

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

				cnn_convolutional_layers.push_back(kernels_rearr);

				// Place-holders for DFT precomputation
				std::vector<std::map<int, std::vector<cv::Mat_<double> > > > cnn_convolutional_layers_dft_curr_layer;
				cnn_convolutional_layers_dft_curr_layer.resize(num_kernels);
				cnn_convolutional_layers_dft.push_back(cnn_convolutional_layers_dft_curr_layer);

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

				cnn_convolutional_layers_weights.push_back(W.t());
				conv_layer_pre_alloc_im2col.push_back(cv::Mat_<float>());

			}
			else if (layer_type == 1)
			{
				int kernel_x, kernel_y, stride_x, stride_y;
				cnn_stream.read((char*)&kernel_x, 4);
				cnn_stream.read((char*)&kernel_y, 4);
				cnn_stream.read((char*)&stride_x, 4);
				cnn_stream.read((char*)&stride_y, 4);
				cnn_max_pooling_layers.push_back(std::tuple<int, int, int, int>(kernel_x, kernel_y, stride_x, stride_y));
			}
			else if (layer_type == 2)
			{
				cv::Mat_<float> biases;
				ReadMatBin(cnn_stream, biases);
				cnn_fully_connected_layers_biases.push_back(biases);

				// Fully connected layer
				cv::Mat_<float> weights;
				ReadMatBin(cnn_stream, weights);
				cnn_fully_connected_layers_weights.push_back(weights.t());
			}

			else if (layer_type == 3)
			{
				cv::Mat_<float> weights;
				ReadMatBin(cnn_stream, weights);
				cnn_prelu_layer_weights.push_back(weights);
			}
		}
	}
	else
	{
		std::cout << "WARNING: Can't find the CNN location" << std::endl;
	}
}

//===========================================================================
// Read in the MTCNN detector
void FaceDetectorMTCNN::Read(const std::string& location)
{

	//std::cout << "Reading the MTCNN face detector from: " << location << std::endl;

	std::ifstream locations(location.c_str(), std::ios_base::in);
	if (!locations.is_open())
	{
		std::cout << "MTCNN model file not found or can't be opened" << std::endl;
		return;
	}
	std::string line;

	// The other module locations should be defined as relative paths from the main model
	fs::path root = fs::path(location).parent_path();

	// The main file contains the references to other files
	while (!locations.eof())
	{
		getline(locations, line);

		std::stringstream lineStream(line);

		std::string module;
		std::string location;

		// figure out which module is to be read from which file
		lineStream >> module;

		lineStream >> location;

		// remove carriage return at the end for compatibility with unix systems
		if (location.size() > 0 && location.at(location.size() - 1) == '\r')
		{
			location = location.substr(0, location.size() - 1);
		}

		// append to root
		location = (root / location).string();
		if (module.compare("PNet") == 0)
		{
			//std::cout << "Reading the PNet module from: " << location << std::endl;
			PNet.Read(location);
		}
		else if(module.compare("RNet") == 0)
		{
			//std::cout << "Reading the RNet module from: " << location << std::endl;
			RNet.Read(location);
		}
		else if (module.compare("ONet") == 0)
		{
			//std::cout << "Reading the ONet module from: " << location << std::endl;
			ONet.Read(location);
		}
	}
}

// Perform non maximum supression on proposal bounding boxes prioritizing boxes with high score/confidence
std::vector<int> non_maximum_supression(const std::vector<cv::Rect_<float> >& original_bb, const std::vector<float>& scores, float thresh, bool minimum)
{

	// Sort the input bounding boxes by the detection score, using the nice trick of multimap always being sorted internally
	std::multimap<float, size_t> idxs;
	for (size_t i = 0; i < original_bb.size(); ++i)
	{
		idxs.insert(std::pair<float, size_t>(scores[i], i));
	}

	std::vector<int> output_ids;

	// keep looping while some indexes still remain in the indexes list
	while (idxs.size() > 0)
	{
		// grab the last rectangle
		auto lastElem = --std::end(idxs);
		size_t curr_id = lastElem->second;

		const cv::Rect& rect1 = original_bb[curr_id];

		idxs.erase(lastElem);

		// Iterate through remaining bounding boxes and choose which ones to remove
		for (auto pos = std::begin(idxs); pos != std::end(idxs); )
		{
			// grab the current rectangle
			const cv::Rect& rect2 = original_bb[pos->second];

			float intArea = (rect1 & rect2).area();
			float unionArea;
			if (minimum)
			{
				unionArea = cv::min(rect1.area(), rect2.area());
			}
			else 
			{
				unionArea = rect1.area() + rect2.area() - intArea;
			}
			float overlap = intArea / unionArea;

			// Remove the bounding boxes with less confidence but with significant overlap with the current one
			if (overlap > thresh)
			{
				pos = idxs.erase(pos);
			}
			else
			{
				++pos;
			}
		}
		output_ids.push_back(curr_id);

	}

	return output_ids;

}

// Helper function for selecting a subset of bounding boxes based on indices
void select_subset(const std::vector<int>& to_keep, std::vector<cv::Rect_<float> >& bounding_boxes, std::vector<float>& scores,
	std::vector<cv::Rect_<float> >& corrections)
{
	std::vector<cv::Rect_<float> > bounding_boxes_tmp;
	std::vector<float> scores_tmp;
	std::vector<cv::Rect_<float> > corrections_tmp;

	for (size_t i = 0; i < to_keep.size(); ++i)
	{
		bounding_boxes_tmp.push_back(bounding_boxes[to_keep[i]]);
		scores_tmp.push_back(scores[to_keep[i]]);
		corrections_tmp.push_back(corrections[to_keep[i]]);
	}
	
	bounding_boxes = bounding_boxes_tmp;
	scores = scores_tmp;
	corrections = corrections_tmp;
}

// Use the heatmap generated by PNet to generate bounding boxes in the original image space, also generate the correction values and scores of the bounding boxes as well
void generate_bounding_boxes(std::vector<cv::Rect_<float> >& o_bounding_boxes, std::vector<float>& o_scores, 
	std::vector<cv::Rect_<float> >& o_corrections, const cv::Mat_<float>& heatmap, const std::vector<cv::Mat_<float> >& corrections,
	float scale, float threshold, int face_support)
{

	// Correction for the pooling
	int stride = 2;

	o_bounding_boxes.clear();
	o_scores.clear();
	o_corrections.clear();

	int counter = 0;
	for (int x = 0; x < heatmap.cols; ++x)
	{
		for(int y = 0; y < heatmap.rows; ++y)
		{
			if (heatmap.at<float>(y, x) >= threshold)
			{
				float min_x = int((stride * x + 1) / scale);
				float max_x = int((stride * x + face_support) / scale);
				float min_y = int((stride * y + 1) / scale);
				float max_y = int((stride * y + face_support) / scale);

				o_bounding_boxes.push_back(cv::Rect_<float>(min_x, min_y, max_x - min_x, max_y - min_y));
				o_scores.push_back(heatmap.at<float>(y, x));

				float corr_x = corrections[0].at<float>(y, x);
				float corr_y = corrections[1].at<float>(y, x);
				float corr_width = corrections[2].at<float>(y, x);
				float corr_height = corrections[3].at<float>(y, x);
				o_corrections.push_back(cv::Rect_<float>(corr_x, corr_y, corr_width, corr_height));

				counter++;
			}
		}
	}
	
}

// Converting the bounding boxes to squares
void rectify(std::vector<cv::Rect_<float> >& total_bboxes)
{

	// Apply size and location offsets
	for (size_t i = 0; i < total_bboxes.size(); ++i)
	{
		float height = total_bboxes[i].height;
		float width = total_bboxes[i].width;

		float max_side = std::max(width, height);

		// Correct the starts based on new size
		float new_min_x = total_bboxes[i].x + 0.5 * (width - max_side);
		float new_min_y = total_bboxes[i].y + 0.5 * (height - max_side);

		total_bboxes[i].x = (int)new_min_x;
		total_bboxes[i].y = (int)new_min_y;
		total_bboxes[i].width = (int)max_side;
		total_bboxes[i].height = (int)max_side;
	}
}

void apply_correction(std::vector<cv::Rect_<float> >& total_bboxes, const std::vector<cv::Rect_<float> > corrections, bool add1)
{

	// Apply size and location offsets
	for (size_t i = 0; i < total_bboxes.size(); ++i)
	{
		cv::Rect curr_box = total_bboxes[i];
		if (add1)
		{
			curr_box.width++;
			curr_box.height++;
		}

		float new_min_x = curr_box.x + corrections[i].x * curr_box.width;
		float new_min_y = curr_box.y + corrections[i].y * curr_box.height;
		float new_max_x = curr_box.x + curr_box.width + curr_box.width * corrections[i].width;
		float new_max_y = curr_box.y + curr_box.height + curr_box.height * corrections[i].height;
		total_bboxes[i] = cv::Rect_<float>(new_min_x, new_min_y, new_max_x - new_min_x, new_max_y - new_min_y);

	}


}


// The actual MTCNN face detection step
bool FaceDetectorMTCNN::DetectFaces(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat& img_in, 
	std::vector<float>& o_confidences, int min_face_size, float t1, float t2, float t3)
{

	int height_orig = img_in.size().height;
	int width_orig = img_in.size().width;

	// Size ratio of image pyramids
	double pyramid_factor = 0.709;

	// Face support region is 12x12 px, so from that can work out the largest
	// scale(which is 12 / min), and work down from there to smallest scale(no smaller than 12x12px)
	int min_dim = std::min(height_orig, width_orig);

	int face_support = 12;
	int num_scales = floor(log((double)min_face_size / (double)min_dim) / log(pyramid_factor)) + 1;

	cv::Mat input_img;

	// Force the image to three channels
	if (img_in.channels() == 1)
	{
		cv::cvtColor(img_in, input_img, cv::COLOR_GRAY2RGB);
	}
	else
	{
		input_img = img_in;
	}

	cv::Mat img_float;
	input_img.convertTo(img_float, CV_32FC3);

	std::vector<cv::Rect_<float> > proposal_boxes_all;
	std::vector<float> scores_all;
	std::vector<cv::Rect_<float> > proposal_corrections_all;

	// As the scales will be done in parallel have some containers for them
	std::vector<std::vector<cv::Rect_<float> > > proposal_boxes_cross_scale(num_scales);
	std::vector<std::vector<float> > scores_cross_scale(num_scales);
	std::vector<std::vector<cv::Rect_<float> > > proposal_corrections_cross_scale(num_scales);

	for (int i = 0; i < num_scales; ++i)
	{
		double scale = ((double)face_support / (double)min_face_size)*cv::pow(pyramid_factor, i);

		int h_pyr = ceil(height_orig * scale);
		int w_pyr = ceil(width_orig * scale);

		cv::Mat normalised_img;
		cv::resize(img_float, normalised_img, cv::Size(w_pyr, h_pyr));
		
		// Normalize the image
		normalised_img = (normalised_img - 127.5) * 0.0078125;

		// Actual PNet CNN step
		std::vector<cv::Mat_<float> > pnet_out = PNet.Inference(normalised_img, true, false);

		// Clear the precomputations, as the image sizes will be different
		PNet.ClearPrecomp();

		// Extract the probabilities from PNet response
		cv::Mat_<float> prob_heatmap;
		cv::exp(pnet_out[0]- pnet_out[1], prob_heatmap);
		prob_heatmap = 1.0 / (1.0 + prob_heatmap);

		// Extract the probabilities from PNet response
		std::vector<cv::Mat_<float>> corrections_heatmap(pnet_out.begin() + 2, pnet_out.end());

		// Grab the detections
		std::vector<cv::Rect_<float> > proposal_boxes;
		std::vector<float> scores;
		std::vector<cv::Rect_<float> > proposal_corrections;
		generate_bounding_boxes(proposal_boxes, scores, proposal_corrections, prob_heatmap, corrections_heatmap, scale, t1, face_support);

		proposal_boxes_cross_scale[i] = proposal_boxes;
		scores_cross_scale[i] = scores;
		proposal_corrections_cross_scale[i] = proposal_corrections;
	}
	//});

	// Perform non-maximum supression on proposals accross scales and combine them
	for (int i = 0; i < num_scales; ++i)
	{
		std::vector<int> to_keep = non_maximum_supression(proposal_boxes_cross_scale[i], scores_cross_scale[i], 0.5, false);
		select_subset(to_keep, proposal_boxes_cross_scale[i], scores_cross_scale[i], proposal_corrections_cross_scale[i]);

		proposal_boxes_all.insert(proposal_boxes_all.end(), proposal_boxes_cross_scale[i].begin(), proposal_boxes_cross_scale[i].end());
		scores_all.insert(scores_all.end(), scores_cross_scale[i].begin(), scores_cross_scale[i].end());
		proposal_corrections_all.insert(proposal_corrections_all.end(), proposal_corrections_cross_scale[i].begin(), proposal_corrections_cross_scale[i].end());
	}

	// Preparation for RNet step

	// Non maximum supression accross bounding boxes, and their offset correction
	std::vector<int> to_keep = non_maximum_supression(proposal_boxes_all, scores_all, 0.7, false);
	select_subset(to_keep, proposal_boxes_all, scores_all, proposal_corrections_all);

	apply_correction(proposal_boxes_all, proposal_corrections_all, false);

	// Convert to rectangles and round
	rectify(proposal_boxes_all);

	// Creating proposal images from previous step detections
	std::vector<bool> above_thresh;
	above_thresh.resize(proposal_boxes_all.size(), false);

	for (size_t k = 0; k < proposal_boxes_all.size(); ++k) 
	{
		float width_target = proposal_boxes_all[k].width + 1;
		float height_target = proposal_boxes_all[k].height + 1;

		// Work out the start and end indices in the original image
		int start_x_in = cv::max((int)(proposal_boxes_all[k].x - 1), 0);
		int start_y_in = cv::max((int)(proposal_boxes_all[k].y - 1), 0);
		int end_x_in = cv::min((int)(proposal_boxes_all[k].x + width_target - 1), width_orig);
		int end_y_in = cv::min((int)(proposal_boxes_all[k].y + height_target - 1), height_orig);

		// Work out the start and end indices in the target image
		int	start_x_out = cv::max((int)(-proposal_boxes_all[k].x + 1), 0);
		int start_y_out = cv::max((int)(-proposal_boxes_all[k].y + 1), 0);
		int end_x_out = cv::min(width_target - (proposal_boxes_all[k].x + proposal_boxes_all[k].width - width_orig), width_target);
		int end_y_out = cv::min(height_target - (proposal_boxes_all[k].y + proposal_boxes_all[k].height - height_orig), height_target);

		cv::Mat tmp(height_target, width_target, CV_32FC3, cv::Scalar(0.0f,0.0f,0.0f));

		img_float(cv::Rect(start_x_in, start_y_in, end_x_in - start_x_in, end_y_in - start_y_in)).copyTo(
			tmp(cv::Rect(start_x_out, start_y_out, end_x_out - start_x_out, end_y_out - start_y_out)));
		
		cv::Mat prop_img;
		cv::resize(tmp, prop_img, cv::Size(24, 24));
			
		prop_img = (prop_img - 127.5) * 0.0078125;
		
		// Perform RNet on the proposal image
		std::vector<cv::Mat_<float> > rnet_out = RNet.Inference(prop_img, true, false);

		float prob = 1.0 / (1.0 + cv::exp(rnet_out[0].at<float>(0) - rnet_out[0].at<float>(1)));
		scores_all[k] = prob;
		proposal_corrections_all[k].x = rnet_out[0].at<float>(2);
		proposal_corrections_all[k].y = rnet_out[0].at<float>(3);
		proposal_corrections_all[k].width = rnet_out[0].at<float>(4);
		proposal_corrections_all[k].height = rnet_out[0].at<float>(5);
		if(prob >= t2)
		{
			above_thresh[k] = true;
		}
		else
		{
			above_thresh[k] = false;
		}

	}
	//});

	to_keep.clear();
	for (size_t i = 0; i < above_thresh.size(); ++i)
	{
		if (above_thresh[i])
		{
			to_keep.push_back(i);
		}
	}

	// Pick only the bounding boxes above the threshold
	select_subset(to_keep, proposal_boxes_all, scores_all, proposal_corrections_all);

	// Non maximum supression accross bounding boxes, and their offset correction
	to_keep = non_maximum_supression(proposal_boxes_all, scores_all, 0.7, false);
	select_subset(to_keep, proposal_boxes_all, scores_all, proposal_corrections_all);

	apply_correction(proposal_boxes_all, proposal_corrections_all, false);

	// Convert to rectangles and round
	rectify(proposal_boxes_all);

	// Preparing for the ONet stage
	above_thresh.clear();
	above_thresh.resize(proposal_boxes_all.size());

	for (size_t k = 0; k < proposal_boxes_all.size(); ++k)
	{
		float width_target = proposal_boxes_all[k].width + 1;
		float height_target = proposal_boxes_all[k].height + 1;

		// Work out the start and end indices in the original image
		int start_x_in = cv::max((int)(proposal_boxes_all[k].x - 1), 0);
		int start_y_in = cv::max((int)(proposal_boxes_all[k].y - 1), 0);
		int end_x_in = cv::min((int)(proposal_boxes_all[k].x + width_target - 1), width_orig);
		int end_y_in = cv::min((int)(proposal_boxes_all[k].y + height_target - 1), height_orig);

		// Work out the start and end indices in the target image
		int	start_x_out = cv::max((int)(-proposal_boxes_all[k].x + 1), 0);
		int start_y_out = cv::max((int)(-proposal_boxes_all[k].y + 1), 0);
		int end_x_out = cv::min(width_target - (proposal_boxes_all[k].x + proposal_boxes_all[k].width - width_orig), width_target);
		int end_y_out = cv::min(height_target - (proposal_boxes_all[k].y + proposal_boxes_all[k].height - height_orig), height_target);

		cv::Mat tmp(height_target, width_target, CV_32FC3, cv::Scalar(0.0f, 0.0f, 0.0f));

		img_float(cv::Rect(start_x_in, start_y_in, end_x_in - start_x_in, end_y_in - start_y_in)).copyTo(
			tmp(cv::Rect(start_x_out, start_y_out, end_x_out - start_x_out, end_y_out - start_y_out)));

		cv::Mat prop_img;
		cv::resize(tmp, prop_img, cv::Size(48, 48));

		prop_img = (prop_img - 127.5) * 0.0078125;

		// Perform RNet on the proposal image
		std::vector<cv::Mat_<float> > onet_out = ONet.Inference(prop_img, true, false);

		float prob = 1.0 / (1.0 + cv::exp(onet_out[0].at<float>(0) - onet_out[0].at<float>(1)));
		scores_all[k] = prob;
		proposal_corrections_all[k].x = onet_out[0].at<float>(2);
		proposal_corrections_all[k].y = onet_out[0].at<float>(3);
		proposal_corrections_all[k].width = onet_out[0].at<float>(4);
		proposal_corrections_all[k].height = onet_out[0].at<float>(5);
		if (prob >= t3)
		{
			above_thresh[k] = true;
		}
		else
		{
			above_thresh[k] = false;
		}
	}
	//});

	to_keep.clear();
	for (size_t i = 0; i < above_thresh.size(); ++i)
	{
		if (above_thresh[i])
		{
			to_keep.push_back(i);
		}
	}

	// Pick only the bounding boxes above the threshold
	select_subset(to_keep, proposal_boxes_all, scores_all, proposal_corrections_all);
	apply_correction(proposal_boxes_all, proposal_corrections_all, true);

	// Non maximum supression accross bounding boxes, and their offset correction
	to_keep = non_maximum_supression(proposal_boxes_all, scores_all, 0.7, true);
	select_subset(to_keep, proposal_boxes_all, scores_all, proposal_corrections_all);

	// Correct the box to expectation to be tight around facial landmarks
	for (size_t k = 0; k < proposal_boxes_all.size(); ++k)
	{
		proposal_boxes_all[k].x = proposal_boxes_all[k].width * -0.0075 + proposal_boxes_all[k].x;
		proposal_boxes_all[k].y = proposal_boxes_all[k].height * 0.2459 + proposal_boxes_all[k].y;
		proposal_boxes_all[k].width = 1.0323 * proposal_boxes_all[k].width;
		proposal_boxes_all[k].height = 0.7751 * proposal_boxes_all[k].height;

		o_regions.push_back(cv::Rect_<float>(proposal_boxes_all[k].x, proposal_boxes_all[k].y, proposal_boxes_all[k].width, proposal_boxes_all[k].height));
		o_confidences.push_back(scores_all[k]);

	}

	if(o_regions.size() > 0)
	{
		return true;
	}
	else
	{
		return false;
	}
}

