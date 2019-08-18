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

#include "SequenceCapture.h"
#include "ImageManipulationHelpers.h"

using namespace Utilities;

#define INFO_STREAM( stream ) \
std::cout << stream << std::endl

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

#define ERROR_STREAM( stream ) \
std::cout << "Error: " << stream << std::endl

bool SequenceCapture::Open(std::vector<std::string>& arguments)
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

	std::string input_video_file;
	std::string input_sequence_directory;
	int device = -1;
	int cam_width = 640;
	int cam_height = 480;

	bool file_found = false;

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (!file_found && arguments[i].compare("-f") == 0)
		{
			input_video_file = (input_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
			file_found = true;
		}
		else if (!file_found && arguments[i].compare("-fdir") == 0)
		{
			input_sequence_directory = (input_root + arguments[i + 1]);
			valid[i] = false;
			valid[i + 1] = false;
			i++;
			file_found = true;
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
		else if (arguments[i].compare("-device") == 0)
		{
			std::stringstream data(arguments[i + 1]);
			data >> device;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-cam_width") == 0)
		{
			std::stringstream data(arguments[i + 1]);
			data >> cam_width;
			valid[i] = false;
			valid[i + 1] = false;
			i++;
		}
		else if (arguments[i].compare("-cam_height") == 0)
		{
			std::stringstream data(arguments[i + 1]);
			data >> cam_height;
			valid[i] = false;
			valid[i + 1] = false;
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
	
	no_input_specified = !file_found;

	// Based on what was read in open the sequence
	if (device != -1)
	{
		return OpenWebcam(device, cam_width, cam_height, fx, fy, cx, cy);
	}
	if (!input_video_file.empty())
	{
		return OpenVideoFile(input_video_file, fx, fy, cx, cy);
	}
	if (!input_sequence_directory.empty())
	{
		return OpenImageSequence(input_sequence_directory, fx, fy, cx, cy);
	}

	// If no input found return false and set a flag for it
	no_input_specified = true;

	return false;
}

// Get current date/time, format is YYYY-MM-DD.HH:mm, useful for saving data from webcam
const std::string currentDateTime() 
{

	time_t rawtime;
	struct tm * timeinfo;
	char buffer[80];

	time(&rawtime);
	timeinfo = localtime(&rawtime);

	strftime(buffer, sizeof(buffer), "%Y-%m-%d-%H-%M", timeinfo);

	return buffer;
}


bool SequenceCapture::OpenWebcam(int device, int image_width, int image_height, float fx, float fy, float cx, float cy)
{
	INFO_STREAM("Attempting to read from webcam: " << device);

	no_input_specified = false;
	frame_num = 0;
	time_stamp = 0;

	if (device < 0)
	{
		std::cout << "Specify a valid device" << std::endl;
		return false;
	}

	latest_frame = cv::Mat();
	latest_gray_frame = cv::Mat();

	capture.open(device);
	capture.set(cv::CAP_PROP_FRAME_WIDTH, image_width);
	capture.set(cv::CAP_PROP_FRAME_HEIGHT, image_height);

	is_webcam = true;
	is_image_seq = false;

	vid_length = 0;

	this->frame_width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
	this->frame_height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	if (!capture.isOpened())
	{
		std::cout << "Failed to open the webcam" << std::endl;
		return false;
	}
	if (frame_width != image_width || frame_height != image_height)
	{
		std::cout << "Failed to open the webcam with desired resolution" << std::endl;
		std::cout << "Defaulting to " << frame_width << "x" << frame_height << std::endl;
	}

	this->fps = capture.get(cv::CAP_PROP_FPS);

	// Check if fps is nan or less than 0
	if (fps != fps || fps <= 0)
	{
		INFO_STREAM("FPS of the webcam cannot be determined, assuming 30");
		fps = 30;
	}
	
	SetCameraIntrinsics(fx, fy, cx, cy);
	std::string time = currentDateTime();
	this->name = "webcam_" + time;

	start_time = cv::getTickCount();
	capturing = true;

	return true;

}

void SequenceCapture::Close()
{
	// Close the capturing threads
	capturing = false;

	// If the queue is full it will be blocked, so need to empty it
	while (!capture_queue.empty())
	{
		capture_queue.pop();
	}

	if (capture_thread.joinable())
		capture_thread.join();
	
	// Release the capture objects
	if (capture.isOpened())
		capture.release();

}

// Destructor that releases the capture
SequenceCapture::~SequenceCapture()
{
	Close();
}

bool SequenceCapture::OpenVideoFile(std::string video_file, float fx, float fy, float cx, float cy)
{
	INFO_STREAM("Attempting to read from file: " << video_file);

	no_input_specified = false;
	frame_num = 0;
	time_stamp = 0;

	latest_frame = cv::Mat();
	latest_gray_frame = cv::Mat();

	capture.open(video_file);

	if (!capture.isOpened())
	{
		std::cout << "Failed to open the video file at location: " << video_file << std::endl;
		return false;
	}

	this->fps = capture.get(cv::CAP_PROP_FPS);
	
	// Check if fps is nan or less than 0
	if (fps != fps || fps <= 0)
	{
		WARN_STREAM("FPS of the video file cannot be determined, assuming 30");
		fps = 30;
	}

	is_webcam = false;
	is_image_seq = false;
	
	this->frame_width = (int)capture.get(cv::CAP_PROP_FRAME_WIDTH);
	this->frame_height = (int)capture.get(cv::CAP_PROP_FRAME_HEIGHT);

	vid_length = (int)capture.get(cv::CAP_PROP_FRAME_COUNT);

	SetCameraIntrinsics(fx, fy, cx, cy);

	this->name = video_file;
	capturing = true;

	capture_thread = std::thread(&SequenceCapture::CaptureThread, this);

	return true;

}

bool SequenceCapture::OpenImageSequence(std::string directory, float fx, float fy, float cx, float cy)
{
	INFO_STREAM("Attempting to read from directory: " << directory);

	no_input_specified = false;
	frame_num = 0;
	time_stamp = 0;

	image_files.clear();

	fs::path image_directory(directory);

	if (!fs::exists(image_directory))
	{
		std::cout << "Provided directory does not exist: " << directory << std::endl;
		return false;
	}

	std::vector<fs::path> file_in_directory;
	copy(fs::directory_iterator(image_directory), fs::directory_iterator(), back_inserter(file_in_directory));

	// Sort the images in the directory first
	sort(file_in_directory.begin(), file_in_directory.end());

	std::vector<std::string> curr_dir_files;

	for (std::vector<fs::path>::const_iterator file_iterator(file_in_directory.begin()); file_iterator != file_in_directory.end(); ++file_iterator)
	{
		// Possible image extension .jpg and .png
		if (file_iterator->extension().string().compare(".jpg") == 0 || file_iterator->extension().string().compare(".jpeg") == 0  || file_iterator->extension().string().compare(".png") == 0 || file_iterator->extension().string().compare(".bmp") == 0)
		{
			curr_dir_files.push_back(file_iterator->string());
		}
	}

	image_files = curr_dir_files;

	if (image_files.empty())
	{
		std::cout << "No images found in the directory: " << directory << std::endl;
		return false;
	}

	// Assume all images are same size in an image sequence
	cv::Mat tmp = cv::imread(image_files[0], cv::IMREAD_COLOR);
	this->frame_height = tmp.size().height;
	this->frame_width = tmp.size().width;

	SetCameraIntrinsics(fx, fy, cx, cy);

	// No fps as we have a sequence
	this->fps = 0;

	this->name = directory;

	is_webcam = false;
	is_image_seq = true;	
	vid_length = image_files.size();
	capturing = true;

	capture_thread = std::thread(&SequenceCapture::CaptureThread, this);
	
	return true;

}

void SequenceCapture::SetCameraIntrinsics(float fx, float fy, float cx, float cy)
{
	// If optical centers are not defined just use center of image
	if (cx == -1)
	{
		this->cx = this->frame_width / 2.0f;
		this->cy = this->frame_height / 2.0f;
	}
	else
	{
		this->cx = cx;
		this->cy = cy;
	}
	// Use a rough guess-timate of focal length
	if (fx == -1)
	{
		this->fx = 500.0f * (this->frame_width / 640.0f);
		this->fy = 500.0f * (this->frame_height / 480.0f);

		this->fx = (this->fx + this->fy) / 2.0f;
		this->fy = this->fx;
	}
	else
	{
		this->fx = fx;
		this->fy = fy;
	}
}

void SequenceCapture::CaptureThread()
{
	int capacity = (CAPTURE_CAPACITY * 1024 * 1024) / (4 * frame_width * frame_height);
	capture_queue.set_capacity(capacity);

	int frame_num_int = 0;

	while(capturing)
	{
		double timestamp_curr = 0;
		cv::Mat tmp_frame;
		cv::Mat_<uchar> tmp_gray_frame;

		if (!is_image_seq)
		{
			bool success = capture.read(tmp_frame);

			if (!success)
			{
				// Indicate lack of success by returning an empty image
				tmp_frame = cv::Mat();
				capturing = false;
			}

			// Recording the timestamp
			timestamp_curr = frame_num_int * (1.0 / fps);			
		}
		else if (is_image_seq)
		{
			if (image_files.empty() || frame_num_int >= (int)image_files.size())
			{
				// Indicate lack of success by returning an empty image
				tmp_frame = cv::Mat();
				capturing = false;
			}
			else
			{
				tmp_frame = cv::imread(image_files[frame_num_int], cv::IMREAD_COLOR);
			}
			timestamp_curr = 0;
		}

		frame_num_int++;
		// Set the grayscale frame
		ConvertToGrayscale_8bit(tmp_frame, tmp_gray_frame);

		capture_queue.push(std::make_tuple(timestamp_curr, tmp_frame, tmp_gray_frame));
		
	}
}

cv::Mat SequenceCapture::GetNextFrame()
{
	if(!is_webcam)
	{
		std::tuple<double, cv::Mat, cv::Mat_<uchar> > data;

		data = capture_queue.pop();

		time_stamp = std::get<0>(data);
		latest_frame = std::get<1>(data);
		latest_gray_frame = std::get<2>(data);

	}
	else
	{
		// Webcam does not use the threaded interface
		bool success = capture.read(latest_frame);

		time_stamp = (cv::getTickCount() - start_time) / cv::getTickFrequency();

		if (!success)
		{
			// Indicate lack of success by returning an empty image
			latest_frame = cv::Mat();
		}
		
		ConvertToGrayscale_8bit(latest_frame, latest_gray_frame);

	}
	frame_num++;

	return latest_frame;
}

double SequenceCapture::GetProgress()
{
	if (is_webcam)
	{
		return -1.0;
	}
	else
	{
		return (double)frame_num / (double)vid_length;
	}
}

bool SequenceCapture::IsOpened()
{
	if (is_webcam || !is_image_seq)
		return capture.isOpened();
	else
		return (image_files.size() > 0 && frame_num < image_files.size());
}

cv::Mat_<uchar> SequenceCapture::GetGrayFrame() 
{
	return latest_gray_frame;
}
