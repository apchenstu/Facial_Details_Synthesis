///////////////////////////////////////////////////////////////////////////////
// Copyright (C) 2017, Tadas Baltrusaitis all rights reserved.
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

#ifndef SEQUENCE_CAPTURE_H
#define SEQUENCE_CAPTURE_H

// System includes
#include <fstream>
#include <sstream>
#include <vector>

#include <thread>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <ConcurrentQueue.h>

namespace Utilities
{

	//===========================================================================
	/**
	A class for capturing sequences from video, webcam, and image directories
	*/
	class SequenceCapture {

	public:

		// Default constructor
		SequenceCapture() {};

		// Destructor
		~SequenceCapture();

		// Opening based on command line arguments
		bool Open(std::vector<std::string>& arguments);

		// Direct opening

		// Webcam
		bool OpenWebcam(int device_id, int image_width = 640, int image_height = 480, float fx = -1, float fy = -1, float cx = -1, float cy = -1);

		// Image sequence in the directory
		bool OpenImageSequence(std::string directory, float fx = -1, float fy = -1, float cx = -1, float cy = -1);

		// Video file
		bool OpenVideoFile(std::string video_file, float fx = -1, float fy = -1, float cx = -1, float cy = -1);

		bool IsWebcam() { return is_webcam; }

		// Getting the next frame
		cv::Mat GetNextFrame();

		// Getting the most recent grayscale frame (need to call GetNextFrame first)
		cv::Mat_<uchar> GetGrayFrame();

		// Parameters describing the sequence and it's progress
		double GetProgress();

		size_t GetFrameNumber() { return frame_num; }

		bool IsOpened();

		void Close();

		int frame_width;
		int frame_height;

		float fx, fy, cx, cy;

		double fps;

		double time_stamp;

		// Name of the video file, image directory, or the webcam
		std::string name;

		// Allows to differentiate if failed because no input specified or if failed to open a specified input
		bool no_input_specified;

				// Storing the captured data queue
		static const int CAPTURE_CAPACITY = 200; // 200 MB

	private:

		// For faster input, multi-thread the capture so it is not waiting for processing to be done

		// Used to keep track if the recording is still going (for the writing threads)
		bool capturing;

		// For keeping track of tasks
		std::thread capture_thread;

		// A thread that will write video output, so that the rest of the application does not block on it
		void CaptureThread();

		// Blocking copy and move, as it doesn't make sense to have several readers pointed at the same source, and this would cause issues, especially with webcams
		SequenceCapture & operator= (const SequenceCapture& other);
		SequenceCapture & operator= (const SequenceCapture&& other);
		SequenceCapture(const SequenceCapture&& other);
		SequenceCapture(const SequenceCapture& other);

		// Used for capturing webcam and video
		cv::VideoCapture capture;

		// Storing the latest captures
		cv::Mat latest_frame;
		cv::Mat_<uchar> latest_gray_frame;
		
		// Storing capture timestamp, RGB image, gray image
		ConcurrentQueue<std::tuple<double, cv::Mat, cv::Mat_<uchar> > > capture_queue;

		// Keeping track of frame number and the files in the image sequence
		size_t  frame_num;
		std::vector<std::string> image_files;

		// Length of video allowing to assess progress
		size_t vid_length;

		// If using a webcam, helps to keep track of time
		int64 start_time;

		// Keeping track if we are opening a video, webcam or image sequence
		bool is_webcam;
		bool is_image_seq;

		void SetCameraIntrinsics(float fx, float fy, float cx, float cy);


	};
}
#endif // SEQUENCE_CAPTURE_H