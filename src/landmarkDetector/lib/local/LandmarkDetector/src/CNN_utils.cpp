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

#include "stdafx.h"

#include "CNN_utils.h"

namespace LandmarkDetector
{

	// Parametric ReLU with leaky weights (separate ones per channel)
	void PReLU(std::vector<cv::Mat_<float> >& input_output_maps, cv::Mat_<float> prelu_weights)
	{

		if (input_output_maps.size() > 1)
		{
			unsigned int h = input_output_maps[0].rows;
			unsigned int w = input_output_maps[0].cols;
			unsigned int size_in = h * w;

			for (int k = 0; k < (int) input_output_maps.size(); ++k)
			{
				// Apply the PReLU
				auto iter = input_output_maps[k].begin();

				float neg_mult = prelu_weights.at<float>(k);

				for (unsigned int i = 0; i < size_in; ++i)
				{
					float in_val = *iter;

					// The prelu step
					*iter++ = in_val >= 0 ? in_val : in_val * neg_mult;

				}
			}
		}
		else
		{

			int w = input_output_maps[0].cols;

			for (int k = 0; k < prelu_weights.rows; ++k)
			{
				auto iter = input_output_maps[0].row(k).begin();
				float neg_mult = prelu_weights.at<float>(k);

				for (int i = 0; i < w; ++i)
				{
					float in_val = *iter;
					// Apply the PReLU
					*iter = in_val >= 0 ? in_val : in_val * neg_mult;					

					// To deal with OpenCV 3.4s debug mode not allowing to go over iteration boundaries
					if(i + 1 < w)
					{
						iter++;
					}
				}
			}

		}

	}

	void fully_connected(std::vector<cv::Mat_<float> >& outputs, const std::vector<cv::Mat_<float> >& input_maps, cv::Mat_<float> weights, cv::Mat_<float> biases)
	{
		outputs.clear();

		if (input_maps.size() > 1)
		{
			// Concatenate all the maps
			cv::Size orig_size = input_maps[0].size();
			cv::Mat_<float> input_concat((int)input_maps.size(), input_maps[0].cols * input_maps[0].rows);

			for (int in = 0; in < (int)input_maps.size(); ++in)
			{
				cv::Mat_<float> add = input_maps[in];

				// Reshape if all of the data will be flattened
				if (input_concat.rows != weights.cols)
				{
					add = add.t();
				}

				add = add.reshape(0, 1);
				add.copyTo(input_concat.row(in));
			}

			// Treat the input as separate feature maps
			if (input_concat.rows == weights.cols)
			{
				input_concat = weights * input_concat;
				// Add biases
				for (int k = 0; k < biases.rows; ++k)
				{
					input_concat.row(k) = input_concat.row(k) + biases.at<float>(k);
				}

				outputs.clear();
				// Resize and add as output
				for (int k = 0; k < biases.rows; ++k)
				{
					cv::Mat_<float> reshaped = input_concat.row(k).clone();
					reshaped = reshaped.reshape(1, orig_size.height);
					outputs.push_back(reshaped);
				}
			}
			else
			{
				// Flatten the input
				input_concat = input_concat.reshape(0, input_concat.rows * input_concat.cols);

				input_concat = weights * input_concat + biases;

				outputs.clear();
				outputs.push_back(input_concat);
			}

		}
		else
		{
			cv::Mat out = weights * input_maps[0] + biases;
			outputs.clear();
			outputs.push_back(out.t());
		}

	}


	void max_pooling(std::vector<cv::Mat_<float> >& outputs, const std::vector<cv::Mat_<float> >& input_maps, int stride_x, int stride_y, int kernel_size_x, int kernel_size_y)
	{
		std::vector<cv::Mat_<float> > outputs_sub;

		// Iterate over kernel height and width, based on stride
		for (size_t in = 0; in < input_maps.size(); ++in)
		{
			// Help with rounding up a bit, to match caffe style output
			int out_x = (int)round((float)(input_maps[in].cols - kernel_size_x) / (float)stride_x) + 1;
			int out_y = (int)round((float)(input_maps[in].rows - kernel_size_y) / (float)stride_y) + 1;

			cv::Mat_<float> sub_out(out_y, out_x, 0.0);
			cv::Mat_<float> in_map = input_maps[in];

			for (int x = 0; x < input_maps[in].cols; x += stride_x)
			{
				int max_x = cv::min(input_maps[in].cols, x + kernel_size_x);
				int x_in_out = int(x / stride_x);

				if (x_in_out >= out_x)
					continue;

				for (int y = 0; y < input_maps[in].rows; y += stride_y)
				{
					int y_in_out = int(y / stride_y);

					if (y_in_out >= out_y)
						continue;

					int max_y = cv::min(input_maps[in].rows, y + kernel_size_y);

					float curr_max = -FLT_MAX;

					for (int x_in = x; x_in < max_x; ++x_in)
					{
						for (int y_in = y; y_in < max_y; ++y_in)
						{
							float curr_val = in_map.at<float>(y_in, x_in);
							if (curr_val > curr_max)
							{
								curr_max = curr_val;
							}
						}
					}
					sub_out.at<float>(y_in_out, x_in_out) = curr_max;
				}
			}

			outputs_sub.push_back(sub_out);

		}
		outputs = outputs_sub;

	}

	void convolution_single_kern_fft(const std::vector<cv::Mat_<float> >& input_imgs, std::vector<cv::Mat_<double> >& img_dfts, 
		const std::vector<cv::Mat_<float> >&  _templs, std::map<int, std::vector<cv::Mat_<double> > >& _templ_dfts, cv::Mat_<float>& result)
	{
		// Assume result is defined properly
		if (result.empty())
		{
			cv::Size corrSize(input_imgs[0].cols - _templs[0].cols + 1, input_imgs[0].rows - _templs[0].rows + 1);
			result.create(corrSize);
		}

		// Our model will always be under min block size so can ignore this
		//const double blockScale = 4.5;
		//const int minBlockSize = 256;

		int maxDepth = CV_64F;

		cv::Size dftsize;

		dftsize.width = cv::getOptimalDFTSize(result.cols + _templs[0].cols - 1);
		dftsize.height = cv::getOptimalDFTSize(result.rows + _templs[0].rows - 1);

		// Compute block size
		cv::Size blocksize;
		blocksize.width = dftsize.width - _templs[0].cols + 1;
		blocksize.width = MIN(blocksize.width, result.cols);
		blocksize.height = dftsize.height - _templs[0].rows + 1;
		blocksize.height = MIN(blocksize.height, result.rows);

		std::vector<cv::Mat_<double>> dftTempl;

		// if this has not been precomputed, precompute it, otherwise use it
		if (_templ_dfts.find(dftsize.width) == _templ_dfts.end())
		{
			dftTempl.resize(_templs.size());
			for (size_t k = 0; k < _templs.size(); ++k)
			{
				dftTempl[k].create(dftsize.height, dftsize.width);

				cv::Mat_<float> src = _templs[k];

				cv::Mat_<double> dst(dftTempl[k], cv::Rect(0, 0, dftsize.width, dftsize.height));

				cv::Mat_<double> dst1(dftTempl[k], cv::Rect(0, 0, _templs[k].cols, _templs[k].rows));

				if (dst1.data != src.data)
					src.convertTo(dst1, dst1.depth());

				if (dst.cols > _templs[k].cols)
				{
					cv::Mat_<double> part(dst, cv::Range(0, _templs[k].rows), cv::Range(_templs[k].cols, dst.cols));
					part.setTo(0);
				}

				// Perform DFT of the template
				dft(dst, dst, 0, _templs[k].rows);

			}
			_templ_dfts[dftsize.width] = dftTempl;

		}
		else
		{
			dftTempl = _templ_dfts[dftsize.width];
		}

		cv::Size bsz(std::min(blocksize.width, result.cols), std::min(blocksize.height, result.rows));
		cv::Mat src;

		cv::Mat cdst(result, cv::Rect(0, 0, bsz.width, bsz.height));

		std::vector<cv::Mat_<double> > dftImgs;
		dftImgs.resize(input_imgs.size());

		if (img_dfts.empty())
		{
			for (size_t k = 0; k < input_imgs.size(); ++k)
			{
				dftImgs[k].create(dftsize);
				dftImgs[k].setTo(0.0);

				cv::Size dsz(bsz.width + _templs[k].cols - 1, bsz.height + _templs[k].rows - 1);

				int x2 = std::min(input_imgs[k].cols, dsz.width);
				int y2 = std::min(input_imgs[k].rows, dsz.height);

				cv::Mat src0(input_imgs[k], cv::Range(0, y2), cv::Range(0, x2));
				cv::Mat dst(dftImgs[k], cv::Rect(0, 0, dsz.width, dsz.height));
				cv::Mat dst1(dftImgs[k], cv::Rect(0, 0, x2, y2));

				src = src0;

				if (dst1.data != src.data)
					src.convertTo(dst1, dst1.depth());

				dft(dftImgs[k], dftImgs[k], 0, dsz.height);
				img_dfts.push_back(dftImgs[k].clone());
			}
		}

		cv::Mat_<double> dft_img(img_dfts[0].rows, img_dfts[0].cols, 0.0);
		for (size_t k = 0; k < input_imgs.size(); ++k)
		{
			cv::Mat dftTempl1(dftTempl[k], cv::Rect(0, 0, dftsize.width, dftsize.height));
			if (k == 0)
			{
				cv::mulSpectrums(img_dfts[k], dftTempl1, dft_img, 0, true);
			}
			else
			{
				cv::mulSpectrums(img_dfts[k], dftTempl1, dftImgs[k], 0, true);
				dft_img = dft_img + dftImgs[k];
			}
		}

		cv::dft(dft_img, dft_img, cv::DFT_INVERSE + cv::DFT_SCALE, bsz.height);

		src = dft_img(cv::Rect(0, 0, bsz.width, bsz.height));

		src.convertTo(cdst, CV_32F);

	}

	void convolution_fft2(std::vector<cv::Mat_<float> >& outputs, const std::vector<cv::Mat_<float> >& input_maps,
		const std::vector<std::vector<cv::Mat_<float> > >& kernels, const std::vector<float >& biases,
		std::vector<std::map<int, std::vector<cv::Mat_<double> > > >& precomp_dfts)
	{
		outputs.clear();

		// Useful precomputed data placeholders for quick correlation (convolution)
		std::vector<cv::Mat_<double> > input_image_dft;

		for (size_t k = 0; k < kernels.size(); ++k)
		{

			// The convolution (with precomputation)
			cv::Mat_<float> output;
			convolution_single_kern_fft(input_maps, input_image_dft, kernels[k], precomp_dfts[k], output);

			// Combining the maps
			outputs.push_back(output + biases[k]);

		}
	}

	void im2col_t(const cv::Mat_<float>& input, const unsigned int width, const unsigned int height, cv::Mat_<float>& output)
	{

		const unsigned int m = input.cols;
		const unsigned int n = input.rows;

		// determine how many blocks there will be with a sliding window of width x height in the input
		const unsigned int yB = m - height + 1;
		const unsigned int xB = n - width + 1;

		// Allocate the output size
		if (output.rows != width * height || output.cols != xB*yB)
		{
			output = cv::Mat::ones(width * height, xB*yB, CV_32F);
		}

		// Iterate over the whole image
		for (unsigned int i = 0; i< yB; i++)
		{
			unsigned int rowIdx = i;
			for (unsigned int j = 0; j< xB; j++)
			{
				//int rowIdx = i; +j*yB;
				// iterate over the blocks within the image
				for (unsigned int yy = 0; yy < height; ++yy)
				{
					// Faster iteration over the image
					const float* Mi = input.ptr<float>(j + yy);
					for (unsigned int xx = 0; xx < width; ++xx)
					{
						unsigned int colIdx = xx*height + yy;

						output.at<float>(colIdx, rowIdx) = Mi[i + xx];
					}
				}
				rowIdx += yB;

			}
		}
	}

	void im2col(const cv::Mat_<float>& input, const unsigned int width, const unsigned int height, cv::Mat_<float>& output)
	{
	
		const unsigned int m = input.rows;
		const unsigned int n = input.cols;
	
		// determine how many blocks there will be with a sliding window of width x height in the input
		const unsigned int yB = m - height + 1;
		const unsigned int xB = n - width + 1;
	
		// Allocate the output size
		if (output.cols != width * height || output.rows != xB*yB)
		{
			output = cv::Mat::ones(xB*yB, width * height, CV_32F);
		}
	
		// Iterate over the whole image
		for (unsigned int i = 0; i< yB; i++)
		{
			unsigned int rowIdx = i*xB;
			for (unsigned int j = 0; j< xB; j++)
			{
	
				float* Mo = output.ptr<float>(rowIdx);
	
				// iterate over the blocks within the image
				for (unsigned int yy = 0; yy < height; ++yy)
				{
					// Faster iteration over the image
					const float* Mi = input.ptr<float>(i + yy);
	
					for (unsigned int xx = 0; xx < width; ++xx)
					{
						unsigned int colIdx = xx*height + yy;
						//output.at<float>(rowIdx, colIdx) = Mi[j + xx]; //input.at<float>(i + yy, j + xx);
						Mo[colIdx] = Mi[j + xx];
					}
				}
				rowIdx++;
	
			}
		}
	}

	void im2col_multimap(const std::vector<cv::Mat_<float> >& inputs, const unsigned int width, const unsigned int height, 
		cv::Mat_<float>& output)
	{
	
		const unsigned int m = inputs[0].rows;
		const unsigned int n = inputs[0].cols;
	
		// determine how many blocks there will be with a sliding window of width x height in the input
		const unsigned int yB = m - height + 1;
		const unsigned int xB = n - width + 1;
	
		int stride = height * width;
	
		unsigned int num_maps = (unsigned int)inputs.size();
	
		// Allocate the output size
		if (output.cols != width * height * inputs.size() + 1 || (unsigned int) output.rows < xB*yB)
		{
			output = cv::Mat::ones(xB*yB, width * height * num_maps + 1, CV_32F);
		}
	
		// Iterate over the whole image
		for (unsigned int i = 0; i< yB; i++)
		{
			unsigned int rowIdx = i*xB;
			for (unsigned int j = 0; j< xB; j++)
			{
	
				float* Mo = output.ptr<float>(rowIdx);

				// TODO, this should be rearranged and done through mem-copy

				// iterate over the blocks within the image
				for (unsigned int yy = 0; yy < height; ++yy)
				{
					for (unsigned int in_maps = 0; in_maps < num_maps; ++in_maps)
					{
						// Faster iteration over the image
						const float* Mi = inputs[in_maps].ptr<float>(i + yy);
	
						for (unsigned int xx = 0; xx < width; ++xx)
						{
							unsigned int colIdx = xx*height + yy + in_maps * stride;
							//output.at<float>(rowIdx, colIdx) = Mi[j + xx]; //input.at<float>(i + yy, j + xx);
							Mo[colIdx] = Mi[j + xx];
						}
					}
				}
				rowIdx++;
	
			}
		}
	}

	// A fast convolution implementation, can provide a pre-allocated im2col as well, if empty, it is created
	void convolution_direct_blas(std::vector<cv::Mat_<float> >& outputs, const std::vector<cv::Mat_<float> >& input_maps, const cv::Mat_<float>& weight_matrix, int height_k, int width_k, cv::Mat_<float>& pre_alloc_im2col)
	{
		outputs.clear();
	
		int height_in = input_maps[0].rows;
		int width_n = input_maps[0].cols;
	
		// determine how many blocks there will be with a sliding window of width x height in the input
		int yB = height_in - height_k + 1;
		int xB = width_n - width_k + 1;
		int num_rows = yB * xB;

		// Instead of re-allocating data use the first rows of already allocated data and re-allocate only if not enough rows are present, this is what makes this non thread safe, as same memory would be used
		im2col_multimap(input_maps, width_k, height_k, pre_alloc_im2col);
		
		float* m1 = (float*)pre_alloc_im2col.data;
		float* m2 = (float*)weight_matrix.data;
		int m2_cols = weight_matrix.cols;
		int m2_rows = weight_matrix.rows;

		cv::Mat_<float> out(num_rows, weight_matrix.cols, 1.0);
		float* m3 = (float*)out.data;
		
		//cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, weight_t.cols, yB * xB, pre_alloc_im2col.cols, 1, m2, weight_t.cols, m1, pre_alloc_im2col.cols, 0.0, m3, weight_t.cols);
		float alpha = 1.0f;
		float beta = 0.0f;
		// Call fortran directly (faster)
		char N[2]; N[0] = 'N';
		sgemm_(N, N, &m2_cols, &num_rows, &pre_alloc_im2col.cols, &alpha, m2, &m2_cols, m1, &pre_alloc_im2col.cols, &beta, m3, &m2_cols);

		// Above is equivalent to out = pre_alloc_im2col * weight_matrix;
		
		out = out.t();

		// Move back to vectors and reshape accordingly
		for (int k = 0; k < out.rows; ++k)
		{
			outputs.push_back(out.row(k).reshape(1, yB));
		}
	
	}


}
