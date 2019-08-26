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
#include "stdafx_ut.h"

#include "RecorderHOG.h"

using namespace Utilities;

// Default constructor initializes the variables
RecorderHOG::RecorderHOG() :hog_file() {};

// Opening the file and preparing the header for it
bool RecorderHOG::Open(std::string output_file_name)
{
	hog_file.open(output_file_name, std::ios_base::out | std::ios_base::binary);

	return hog_file.is_open();
}

void RecorderHOG::Close()
{
	hog_file.close();
}

void RecorderHOG::Write()
{
	hog_file.write((char*)(&num_cols), 4);
	hog_file.write((char*)(&num_rows), 4);
	hog_file.write((char*)(&num_channels), 4);

	// Not the best way to store a bool, but will be much easier to read it
	float good_frame_float;
	if (good_frame)
		good_frame_float = 1;
	else
		good_frame_float = -1;

	hog_file.write((char*)(&good_frame_float), 4);
	if(hog_descriptor.isContinuous())
	{
		cv::Mat_<float> desc;
		hog_descriptor.convertTo(desc, CV_32F);
		hog_file.write((char*)desc.data, 4 * num_cols * num_rows * 31);
	}
	else
	{
		cv::MatConstIterator_<double> descriptor_it = hog_descriptor.begin();

		for (int y = 0; y < num_cols; ++y)
		{
			for (int x = 0; x < num_rows; ++x)
			{
				for (unsigned int o = 0; o < 31; ++o)
				{

					float hog_data = (float)(*descriptor_it++);
					hog_file.write((char*)&hog_data, 4);
				}
			}
		}
	}
}

// Writing to a HOG file
void RecorderHOG::SetObservationHOG(bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows, int num_channels)
{
	this->num_cols = num_cols;
	this->num_rows = num_rows;
	this->num_channels = num_channels;
	this->hog_descriptor = hog_descriptor;
	this->good_frame = good_frame;
}