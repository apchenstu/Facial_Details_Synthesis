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

#include "ImageCapture.h"
#include "ImageManipulationHelpers.h"

using namespace Utilities;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

bool ImageCapture::Open(std::vector<std::string>& arguments)
{

	// Consuming the input arguments
	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	// Some default values
	std::string input_root = "";
	fx = -1; fy = -1; cx = -1; cy = -1;

	std::string separator = std::string(1, fs::path::preferred_separator);

	// First check if there is a root argument (so that videos and input directories could be defined more easily)
	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-root") == 0)
		{
			input_root = arguments[i + 1] + separator;
			i++;
		}
		if (arguments[i].compare("-inroot") == 0)
		{
			input_root = arguments[i + 1] + separator;
			i++;
		}
	}

	std::string input_directory;
	std::string bbox_directory;

	bool directory_found = false;
	has_bounding_boxes = false;

	std::vector<std::string> input_image_files;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-f") == 0)
		{
			input_image_files.push_back(input_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-fdir") == 0)
		{
			if (directory_found)
			{
				WARN_STREAM("Input directory already found, using the first one:" + input_directory);
			}
			else 
			{
				input_directory = (input_root + arguments[i + 1]);
				valid[i] = false;
				valid[i + 1] = false;
				i++;
				directory_found = true;
			}
		}
		else if (arguments[i].compare("-bboxdir") == 0)
		{
			bbox_directory = (input_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			has_bounding_boxes = true;
			i++;
		}
		else if (arguments[i].compare("-fx") == 0)
		{
			std::stringstream data(arguments[i + 1]);
			data >> fx;
			i++;
		}
		else if (arguments[i].compare("-fy") == 0)
		{
			std::stringstream data(arguments[i + 1]);
			data >> fy;
			i++;
		}
		else if (arguments[i].compare("-cx") == 0)
		{
			std::stringstream data(arguments[i + 1]);
			data >> cx;
			i++;
		}
		else if (arguments[i].compare("-cy") == 0)
		{
			std::stringstream data(arguments[i + 1]);
			data >> cy;
			i++;
		}
	}

	for (int i = (int)arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

	// Based on what was read in open the sequence
	if (!input_image_files.empty())
	{
		return OpenImageFiles(input_image_files, fx, fy, cx, cy);
	}
	if (!input_directory.empty())
	{
		return OpenDirectory(input_directory, bbox_directory, fx, fy, cx, cy);
	}

	// If no input found return false and set a flag for it
	no_input_specified = true;

	return false;
}

bool ImageCapture::OpenImageFiles(const std::vector<std::string>& image_files, float fx, float fy, float cx, float cy)
{
	// Setting some defaults
	frame_num = 0;
	no_input_specified = false;

	latest_frame = cv::Mat();
	latest_gray_frame = cv::Mat();
	this->image_files = image_files;

	// Allow for setting the camera intrinsics, but have to be the same ones for every image
	if (fx != -1 && fy != -1 )
	{
		image_focal_length_set = true;
		this->fx = fx;
		this->fy = fy;

	}
	else
	{
		image_focal_length_set = false;
	}

	if (cx != -1 && cy != -1)
	{
		this->cx = cx;
		this->cy = cy;
		image_optical_center_set = true;
	}
	else
	{
		image_optical_center_set = false;
	}

	return true;

}

bool ImageCapture::OpenDirectory(std::string directory, std::string bbox_directory, float fx, float fy, float cx, float cy)
{
	INFO_STREAM("Attempting to read from directory: " << directory);

	// Setup some defaults
	frame_num = 0;
	no_input_specified = false;

	image_files.clear();

	fs::path image_directory(directory);
	std::vector<fs::path> file_in_directory;
	copy(fs::directory_iterator(image_directory), fs::directory_iterator(), back_inserter(file_in_directory));

	// Sort the images in the directory first
	sort(file_in_directory.begin(), file_in_directory.end());

	std::vector<std::string> curr_dir_files;

	for (std::vector<fs::path>::const_iterator file_iterator(file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
	{
		// Possible image extension .jpg and .png
		if (file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".jpeg") == 0 || file_iterator->extension().string().compare(".png") == 0 || file_iterator->extension().string().compare(".bmp") == 0)
		{
			curr_dir_files.push_back(file_iterator->string());

			// If bounding box directory is specified, read the bounding boxes from it
			if (!bbox_directory.empty())
			{
				fs::path current_file = *file_iterator;
				fs::path bbox_file = bbox_directory / current_file.filename().replace_extension("txt");
				
				// If there is a bounding box file push it to the list of bounding boxes
				if (fs::exists(bbox_file))
				{
					std::ifstream in_bbox(bbox_file.string().c_str(), std::ios_base::in);

					std::vector<cv::Rect_<float> > bboxes_image;

					// Keep reading bounding boxes from a file, stop if empty line or 
					while (!in_bbox.eof())
					{
						std::string bbox_string;
						std::getline(in_bbox, bbox_string);

						if (bbox_string.empty())
							continue;

						std::stringstream ss(bbox_string);

						float min_x, min_y, max_x, max_y;

						ss >> min_x >> min_y >> max_x >> max_y;
						bboxes_image.push_back(cv::Rect_<float>(min_x, min_y, max_x - min_x, max_y - min_y));
					}
					in_bbox.close();

					bounding_boxes.push_back(bboxes_image);
				}
				else
				{
					ERROR_STREAM("Could not find the corresponding bounding box for file:" + file_iterator->string());
					exit(1);
				}
			}
		}
	}

	image_files = curr_dir_files;

	if (image_files.empty())
	{
		std::cout << "No images found in the directory: " << directory << std::endl;
		return false;
	}

	// Allow for setting the camera intrinsics, but have to be the same ones for every image
	if (fx != -1 && fy != -1)
	{
		image_focal_length_set = true;
		this->fx = fx;
		this->fy = fy;

	}
	else
	{
		image_focal_length_set = false;
	}

	if (cx != -1 && cy != -1)
	{
		this->cx = cx;
		this->cy = cy;
		image_optical_center_set = true;
	}
	else
	{
		image_optical_center_set = false;
	}

	return true;

}

void ImageCapture::SetCameraIntrinsics(float fx, float fy, float cx, float cy)
{
	// If optical centers are not defined just use center of image
	if (cx == -1)
	{
		this->cx = this->image_width / 2.0f;
		this->cy = this->image_height / 2.0f;
	}
	else
	{
		this->cx = cx;
		this->cy = cy;
	}
	// Use a rough guess-timate of focal length
	if (fx == -1)
	{
		this->fx = 500.0f * (this->image_width / 640.0f);
		this->fy = 500.0f * (this->image_height / 480.0f);

		this->fx = (this->fx + this->fy) / 2.0f;
		this->fy = this->fx;
	}
	else
	{
		this->fx = fx;
		this->fy = fy;
	}
}

// Returns a read image in 3 channel RGB format, also prepares a grayscale frame if needed
cv::Mat ImageCapture::GetNextImage()
{
	if (image_files.empty() || frame_num >= image_files.size())
	{
		// Indicate lack of success by returning an empty image
		latest_frame = cv::Mat();
		return latest_frame;
	}
		
	// Load the image as an 8 bit RGB
	latest_frame = cv::imread(image_files[frame_num], cv::IMREAD_COLOR);

	if (latest_frame.empty())
	{
		ERROR_STREAM("Could not open the image: " + image_files[frame_num]);
		exit(1);
	}

	image_height = latest_frame.size().height;
	image_width = latest_frame.size().width;

	// Reset the intrinsics for every image if they are not set globally
	float _fx = -1;
	float _fy = -1;
	
	if (image_focal_length_set)
	{
		_fx = fx;
		_fy = fy;
	}
	
	float _cx = -1;
	float _cy = -1;

	if (image_optical_center_set)
	{
		_cx = cx;
		_cy = cy;
	}

	SetCameraIntrinsics(_fx, _fy, _cx, _cy);

	// Set the grayscale frame
	ConvertToGrayscale_8bit(latest_frame, latest_gray_frame);

	this->name = image_files[frame_num];

	frame_num++;

	return latest_frame;
}

std::vector<cv::Rect_<float> > ImageCapture::GetBoundingBoxes()
{
	if (!bounding_boxes.empty())
	{
		return bounding_boxes[frame_num - 1];
	}
	else
	{
		return std::vector<cv::Rect_<float> >();
	}
}

double ImageCapture::GetProgress()
{
	return (double)frame_num / (double)image_files.size();
}

cv::Mat_<uchar> ImageCapture::GetGrayFrame()
{
	return latest_gray_frame;
}
