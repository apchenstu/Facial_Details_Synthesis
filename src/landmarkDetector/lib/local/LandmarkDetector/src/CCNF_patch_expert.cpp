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

#include "CCNF_patch_expert.h"

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// Local includes
#include "LandmarkDetectorUtils.h"

using namespace LandmarkDetector;

// Copy constructors of neuron and patch expert
CCNF_neuron::CCNF_neuron(const CCNF_neuron& other) : weights(other.weights.clone())
{
	this->neuron_type = other.neuron_type;
	this->norm_weights = other.norm_weights;
	this->bias = other.bias;
	this->alpha = other.alpha;

	for (std::map<int, cv::Mat_<double> >::const_iterator it = other.weights_dfts.begin(); it != other.weights_dfts.end(); it++)
	{
		// Make sure the matrix is copied.
		this->weights_dfts.insert(std::pair<int, cv::Mat>(it->first, it->second.clone()));
	}
}

// Copy constructor		
CCNF_patch_expert::CCNF_patch_expert(const CCNF_patch_expert& other) : neurons(other.neurons), window_sizes(other.window_sizes), betas(other.betas)
{
	this->width = other.width;
	this->height = other.height;
	this->patch_confidence = other.patch_confidence;
	
	this->weight_matrix = other.weight_matrix.clone();

	// Copy the Sigmas in a deep way
	for (std::vector<cv::Mat_<float> >::const_iterator it = other.Sigmas.begin(); it != other.Sigmas.end(); it++)
	{
		// Make sure the matrix is copied.
		this->Sigmas.push_back(it->clone());
	}

}

// Compute sigmas for all landmarks for a particular view and window size
void CCNF_patch_expert::ComputeSigmas(std::vector<cv::Mat_<float> > sigma_components, int window_size)
{
	for(size_t i=0; i < window_sizes.size(); ++i)
	{
		if( window_sizes[i] == window_size)
			return;
	}
	// Each of the landmarks will have the same connections, hence constant number of sigma components
	int n_betas = sigma_components.size();

	// calculate the sigmas based on alphas and betas
	float sum_alphas = 0;

	int n_alphas = this->neurons.size();

	// sum the alphas first
	for(int a = 0; a < n_alphas; ++a)
	{
		sum_alphas = sum_alphas + this->neurons[a].alpha;
	}

	cv::Mat_<float> q1 = sum_alphas * cv::Mat_<float>::eye(window_size*window_size, window_size*window_size);

	cv::Mat_<float> q2 = cv::Mat_<float>::zeros(window_size*window_size, window_size*window_size);
	for (int b=0; b < n_betas; ++b)
	{			
		q2 = q2 + ((float)this->betas[b]) * sigma_components[b];
	}

	cv::Mat_<float> SigmaInv = 2 * (q1 + q2);
	
	cv::Mat Sigma_f;
	cv::invert(SigmaInv, Sigma_f, cv::DECOMP_CHOLESKY);

	window_sizes.push_back(window_size);
	Sigmas.push_back(Sigma_f);

}

//===========================================================================
void CCNF_neuron::Read(std::ifstream &stream)
{
	// Sanity check
	int read_type;
	stream.read ((char*)&read_type, 4);
	assert(read_type == 2);

	stream.read ((char*)&neuron_type, 4);
	stream.read ((char*)&norm_weights, 8);
	stream.read ((char*)&bias, 8);
	stream.read ((char*)&alpha, 8);
	
	LandmarkDetector::ReadMatBin(stream, weights); 

}

// Perform im2col, while at the same time doing contrast normalization and adding a bias term 
void im2colContrastNormBias(const cv::Mat_<float>& input, const unsigned int width, const unsigned int height, cv::Mat_<float>& output)
{
	const unsigned int m = input.rows;
	const unsigned int n = input.cols;

	// determine how many blocks there will be with a sliding window of width x height in the input
	const unsigned int yB = m - height + 1;
	const unsigned int xB = n - width + 1;

	// Allocate the output size
	if (output.rows != xB*yB && output.cols != width * height + 1) 
	{
		output = cv::Mat::ones(xB*yB, width * height + 1, CV_32F);
	}

	// Iterate over the blocks
	unsigned int rowIdx = 0;
	for (unsigned int j = 0; j< xB; j++)
	{
		for (unsigned int i = 0; i< yB; i++)
		{

			float* Mo = output.ptr<float>(rowIdx);

			float sum = 0;

			for (unsigned int yy = 0; yy < height; ++yy)
			{
				const float* Mi = input.ptr<float>(i + yy);
				for (unsigned int xx = 0; xx < width; ++xx)
				{
					unsigned int colIdx = xx*height + yy;
					float in = Mi[j + xx];
					sum += in;

					Mo[colIdx + 1] = in;
				}
			}

			// Working out the mean
			float mean = sum / (float)(width * height);

			float sum_sq = 0;
			const unsigned int num_items = width*height + 1;
			// Working out the sum squared and subtracting the mean
			for (unsigned int x = 1; x < num_items; ++x)
			{
				float in = Mo[x] - mean;
				Mo[x] = in;
				sum_sq += in * in;
			}

			float norm = sqrt(sum_sq);

			// Avoiding division by 0
			if (norm == 0)
			{
				norm = 1;
			}

			// Flip multiplication to division for speed
			norm = 1.0 / norm;

			for (unsigned int x = 1; x < num_items; ++x)
			{
				Mo[x] *= norm;
			}

			rowIdx++;
		}
	}
}

//===========================================================================
void CCNF_neuron::Response(const cv::Mat_<float> &im, cv::Mat_<double> &im_dft, cv::Mat &integral_img, cv::Mat &integral_img_sq, cv::Mat_<float> &resp)
{

	int h = im.rows - weights.rows + 1;
	int w = im.cols - weights.cols + 1;
	
	// the patch area on which we will calculate reponses
	cv::Mat_<float> I;

	if(neuron_type == 3)
	{
		// Perform normalisation across whole patch (ignoring the invalid values indicated by <= 0

		cv::Scalar mean;
		cv::Scalar std;
		
		// ignore missing values
		cv::Mat_<uchar> mask = im > 0;
		cv::meanStdDev(im, mean, std, mask);

		// if all values the same don't divide by 0
		if(std[0] != 0)
		{
			I = (im - mean[0]) / std[0];
		}
		else
		{
			I = (im - mean[0]);
		}

		I.setTo(0, mask == 0);
	}
	else
	{
		if(neuron_type == 0)
		{
			I = im;
		}
		else
		{
			printf("ERROR(%s,%d): Unsupported patch type %d!\n", __FILE__,__LINE__,neuron_type);
			abort();
		}
	}
  
	if(resp.empty())
	{		
		resp.create(h, w);
	}

	// The response from neuron before activation
	if(neuron_type == 3)
	{
		// In case of depth we use per area, rather than per patch normalisation
		matchTemplate_m(I, im_dft, integral_img, integral_img_sq, weights, weights_dfts, resp, cv::TM_CCOEFF); // the linear multiplication, efficient calc of response
	}
	else
	{
		matchTemplate_m(I, im_dft, integral_img, integral_img_sq, weights, weights_dfts, resp, cv::TM_CCOEFF_NORMED); // the linear multiplication, efficient calc of response
	}
	
	cv::MatIterator_<float> p = resp.begin();

	cv::MatIterator_<float> q1 = resp.begin(); // respone for each pixel
	cv::MatIterator_<float> q2 = resp.end();

	// the logistic function (sigmoid) applied to the response
	while(q1 != q2)
	{
		*p++ = (2 * alpha) * 1.0 /(1.0 + exp( -(*q1++ * norm_weights + bias )));
	}

}

//===========================================================================
void CCNF_patch_expert::Read(std::ifstream &stream, std::vector<int> window_sizes, std::vector<std::vector<cv::Mat_<float> > > sigma_components)
{

	// Sanity check
	int read_type;

	stream.read ((char*)&read_type, 4);
	assert(read_type == 5);

	// the number of neurons for this patch
	int num_neurons;
	stream.read ((char*)&width, 4);
	stream.read ((char*)&height, 4);
	stream.read ((char*)&num_neurons, 4);

	if(num_neurons == 0)
	{
		// empty patch due to landmark being invisible at that orientation
	
		// read an empty int (due to the way things were written out)
		stream.read ((char*)&num_neurons, 4);
		return;
	}

	neurons.resize(num_neurons);
	for(int i = 0; i < num_neurons; i++)
		neurons[i].Read(stream);
		
	// Combine the neuron weights to one weight matrix for more efficient computation
	weight_matrix = cv::Mat_<float>(neurons.size(), 1 + neurons[0].weights.rows * neurons[0].weights.cols);
	for (size_t i = 0; i < neurons.size(); i++)
	{
		cv::Mat_<float> w_tmp = neurons[i].weights.t();
		cv::Mat_<float> weights_flat = w_tmp.reshape(1, neurons[i].weights.rows * neurons[i].weights.cols);
		weights_flat = weights_flat.t();

		// Incorporate neuron weights directly
		weights_flat = weights_flat * neurons[i].norm_weights;
		weights_flat.copyTo(weight_matrix(cv::Rect(1, i, neurons[i].weights.rows * neurons[i].weights.cols, 1)));
		// Incorporate bias as well
		weight_matrix.at<float>(i, 0) = neurons[i].bias;
	}

	// In case we are using OpenBLAS, make sure it is not multi-threading as we are multi-threading outside of it
	openblas_set_num_threads(1);

	int n_sigmas = window_sizes.size();

	int n_betas = 0;

	if(n_sigmas > 0)
	{
		n_betas = sigma_components[0].size();

		betas.resize(n_betas);

		for (int i=0; i < n_betas;  ++i)
		{
			stream.read ((char*)&betas[i], 8);
		}
	}	

	// Read the patch confidence
	stream.read ((char*)&patch_confidence, 8);

}

//===========================================================================
void CCNF_patch_expert::Response(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response)
{
	
	int response_height = area_of_interest.rows - height + 1;
	int response_width = area_of_interest.cols - width + 1;

	if(response.rows != response_height || response.cols != response_width)
	{
		response.create(response_height, response_width);
	}
		
	response.setTo(0);
	
	// the placeholder for the DFT of the image, the integral image, and squared integral image so they don't get recalculated for every response
	cv::Mat_<double> area_of_interest_dft;
	cv::Mat integral_image, integral_image_sq;
	
	cv::Mat_<float> neuron_response;

	// responses from the neural layers
	for(size_t i = 0; i < neurons.size(); i++)
	{		
		// Do not bother with neuron response if the alpha is tiny and will not contribute much to overall result
		if(neurons[i].alpha > 1e-4)
		{

			neurons[i].Response(area_of_interest, area_of_interest_dft, integral_image, integral_image_sq, neuron_response);
			response = response + neuron_response;
		}
	}

	int s_to_use = -1;

	// Find the matching sigma
	for(size_t i=0; i < window_sizes.size(); ++i)
	{
		if(window_sizes[i] == response_height)
		{
			// Found the correct sigma
			s_to_use = i;			
			break;
		}
	}

	cv::Mat_<float> resp_vec_f = response.reshape(1, response_height * response_width);

	cv::Mat out = Sigmas[s_to_use] * resp_vec_f;
	
	response = out.reshape(1, response_height);

	// Making sure the response does not have negative numbers
	double min;

	minMaxIdx(response, &min, 0);
	if(min < 0)
	{
		response = response - min;
	}

}

void CCNF_patch_expert::ResponseOpenBlas(const cv::Mat_<float> &area_of_interest, cv::Mat_<float> &response, cv::Mat_<float>& im2col_prealloc)
{

	int response_height = area_of_interest.rows - height + 1;
	int response_width = area_of_interest.cols - width + 1;

	if (response.rows != response_height || response.cols != response_width)
	{
		response.create(response_height, response_width);
	}

	response.setTo(0);
	if (neurons.size() == 0)
	{
		return;
	}

	// the placeholder for the column normalized representation of the image, don't get recalculated for every response
	im2colContrastNormBias(area_of_interest, neurons[0].weights.cols, neurons[0].weights.rows, im2col_prealloc);
	cv::Mat_<float> normalized_input = im2col_prealloc.t();

	// the placeholder for the DFT of the image, the integral image, and squared integral image so they don't get recalculated for every response
	cv::Mat_<double> area_of_interest_dft;
	cv::Mat integral_image, integral_image_sq;

	cv::Mat_<float> neuron_response;


	int h = area_of_interest.rows - neurons[0].weights.rows + 1;
	int w = area_of_interest.cols - neurons[0].weights.cols + 1;


	cv::Mat_<float> neuron_resp_full(weight_matrix.rows, normalized_input.cols, 0.0f);
	// Perform matrix multiplication in OpenBLAS (fortran call)
	float alpha1 = 1.0;
	float beta1 = 0.0;
	char N[2]; N[0] = 'N';
	sgemm_(N, N, &normalized_input.cols, &weight_matrix.rows, &weight_matrix.cols, &alpha1, (float*)normalized_input.data, &normalized_input.cols, (float*)weight_matrix.data, &weight_matrix.cols, &beta1, (float*)neuron_resp_full.data, &normalized_input.cols);

	// Above is a faster version of this
	//cv::Mat_<float> neuron_resp_full = this->weight_matrix * normalized_input;

	for (size_t i = 0; i < neurons.size(); i++)
	{
		if (neurons[i].alpha > 1e-4)
		{
			cv::MatIterator_<float> p = response.begin();

			cv::Mat_<float> rel_row = neuron_resp_full.row(i);
			cv::MatIterator_<float> q1 = rel_row.begin(); // respone for each pixel
			cv::MatIterator_<float> q2 = rel_row.end();

			// the logistic function (sigmoid) applied to the response
			while (q1 != q2)
			{
				*p++ += (2.0 * neurons[i].alpha) / (1.0 + exp(-*q1++));
			}
		}
	}
	response = response.t();

	int s_to_use = -1;

	// Find the matching sigma
	for (size_t i = 0; i < window_sizes.size(); ++i)
	{
		if (window_sizes[i] == response_height)
		{
			// Found the correct sigma
			s_to_use = i;
			break;
		}
	}

	cv::Mat_<float> resp_vec_f = response.reshape(1, response_height * response_width);

	cv::Mat_<float> out(Sigmas[s_to_use].rows, resp_vec_f.cols, 0.0f);

	// Perform matrix multiplication in OpenBLAS (fortran call)
	alpha1 = 1.0;
	beta1 = 0.0;
	sgemm_(N, N, &resp_vec_f.cols, &Sigmas[s_to_use].rows, &Sigmas[s_to_use].cols, &alpha1, (float*)resp_vec_f.data, &resp_vec_f.cols, (float*)Sigmas[s_to_use].data, &Sigmas[s_to_use].cols, &beta1, (float*)out.data, &resp_vec_f.cols);

	// Above is a faster version of this
	//cv::Mat out = Sigmas[s_to_use] * resp_vec_f;

	response = out.reshape(1, response_height);

	// Making sure the response does not have negative numbers
	double min;

	minMaxIdx(response, &min, 0);
	if (min < 0)
	{
		response = response - min;
	}

}
