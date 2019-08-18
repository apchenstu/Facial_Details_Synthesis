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
#include <stdafx_fa.h>

#include "Face_utils.h"

#include "SVM_static_lin.h"

using namespace FaceAnalysis;

void SVM_static_lin::Read(std::ifstream& stream, const std::vector<std::string>& au_names)
{

	if(this->means.empty())
	{
		ReadMatBin(stream, this->means);
	}
	else
	{
		cv::Mat_<double> m_tmp;
		ReadMatBin(stream, m_tmp);
		if(cv::norm(m_tmp - this->means > 0.00001))
		{
			std::cout << "Something went wrong with the SVM static classifiers" << std::endl;
		}
	}

	cv::Mat_<double> support_vectors_curr;
	ReadMatBin(stream, support_vectors_curr);

	double bias;
	stream.read((char *)&bias, 8);

	// Read in positive or negative class
	double pos_class;	
	stream.read((char *)&pos_class, 8);

	double neg_class;
	stream.read((char *)&neg_class, 8);


	// Add a column vector to the matrix of support vectors (each column is a support vector)
	if(!this->support_vectors.empty())
	{	
		cv::transpose(this->support_vectors, this->support_vectors);
		cv::transpose(support_vectors_curr, support_vectors_curr);
		this->support_vectors.push_back(support_vectors_curr);
		cv::transpose(this->support_vectors, this->support_vectors);

		cv::transpose(this->biases, this->biases);
		this->biases.push_back(cv::Mat_<double>(1, 1, bias));
		cv::transpose(this->biases, this->biases);

	}
	else
	{
		this->support_vectors.push_back(support_vectors_curr);
		this->biases.push_back(cv::Mat_<double>(1, 1, bias));
	}

	this->pos_classes.push_back(pos_class);
	this->neg_classes.push_back(neg_class);
	
	for(size_t i=0; i < au_names.size(); ++i)
	{
		this->AU_names.push_back(au_names[i]);
	}
}

// Prediction using the HOG descriptor
void SVM_static_lin::Predict(std::vector<double>& predictions, std::vector<std::string>& names, const cv::Mat_<double>& fhog_descriptor, const cv::Mat_<double>& geom_params)
{
	if(AU_names.size() > 0)
	{
		cv::Mat_<double> preds;
		if(fhog_descriptor.cols ==  this->means.cols)
		{
			preds = (fhog_descriptor - this->means) * this->support_vectors + this->biases;
		}
		else
		{
			cv::Mat_<double> input;
			cv::hconcat(fhog_descriptor, geom_params, input);

			preds = (input - this->means) * this->support_vectors + this->biases;
		}

		for(int i = 0; i < preds.cols; ++i)
		{		
			if(preds.at<double>(i) > 0)
			{
				predictions.push_back(pos_classes[i]);
			}
			else
			{
				predictions.push_back(neg_classes[i]);
			}
		}

		names = this->AU_names;
	}
}