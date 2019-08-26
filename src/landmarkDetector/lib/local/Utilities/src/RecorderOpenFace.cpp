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

#include "RecorderOpenFace.h"

using namespace Utilities;

#define WARN_STREAM( stream ) \
std::cout << "Warning: " << stream << std::endl

void CreateDirectory(std::string output_path)
{
	// Removing trailing separators, as that causes issues with directory creation in unix
	while (output_path[output_path.size() - 1] == '/' || output_path[output_path.size() - 1] == '\\')
	{
		output_path = output_path.substr(0, output_path.size() - 1);
	}

	// Creating the right directory structure	
	if (!fs::exists(output_path))
	{
		bool success = fs::create_directories(output_path);

		if (!success)
		{
			std::cout << "ERROR: failed to create output directory:" << output_path << ", do you have permission to create directory" << std::endl;
			exit(1);
		}
	}
}

void RecorderOpenFace::VideoWritingTask(bool is_sequence)
{

	std::pair<std::string, cv::Mat> tracked_data;

	while (true)
	{
		vis_to_out_queue.pop(tracked_data);

		// Indicate that the thread should complete
		if (tracked_data.second.empty())
		{
			break;
		}

		if (is_sequence)
		{
			if (video_writer.isOpened())
			{
				video_writer.write(tracked_data.second);
			}
		}
		else
		{
			bool out_success = cv::imwrite(tracked_data.first, tracked_data.second);
			if (!out_success)
			{
				WARN_STREAM("Could not output tracked image");
			}
		}		
	}
}

void RecorderOpenFace::AlignedImageWritingTask()
{

	std::pair<std::string, cv::Mat> tracked_data;

	while (true)
	{
		aligned_face_queue.pop(tracked_data);

		// Empty frame indicates termination
		if (tracked_data.second.empty())
			break;

		bool write_success = cv::imwrite(tracked_data.first, tracked_data.second);

		if (!write_success)
		{
			WARN_STREAM("Could not output similarity aligned image image");
		}
	}
}

void RecorderOpenFace::PrepareRecording(const std::string& in_filename)
{

	// Construct the directories required for the output
	CreateDirectory(record_root);

	// Create the filename for the general output file that contains all of the meta information about the recording
	fs::path of_det_name(out_name);
	of_det_name = fs::path(record_root) / fs::path(out_name + "_of_details.txt");

	// Write in the of file what we are outputing what is the input etc.
	metadata_file.open(of_det_name.string(), std::ios_base::out);
	if (!metadata_file.is_open())
	{
		std::cout << "ERROR: could not open the output file:" << of_det_name << ", either the path of the output directory is wrong or you do not have the permissions to write to it" << std::endl;
		exit(1);
	}

	// Populate relative and full path names in the meta file, unless it is a webcam
	if (!params.isFromWebcam())
	{
		std::string input_filename_relative = in_filename;
		std::string input_filename_full = in_filename;
		
		if (!fs::path(input_filename_full).is_absolute())
		{
			input_filename_full = fs::canonical(input_filename_relative).string();
		}
		metadata_file << "Input:" << input_filename_relative << std::endl;
		metadata_file << "Input full path:" << input_filename_full << std::endl;
	}
	else
	{
		// Populate the metadata file
		metadata_file << "Input:webcam" << std::endl;
	}

	metadata_file << "Camera parameters:" << params.getFx() << "," << params.getFy() << "," << params.getCx() << "," << params.getCy() << std::endl;

	// Create the required individual recorders, CSV, HOG, aligned, video
	csv_filename = out_name + ".csv";

	// Consruct HOG recorder here
	if (params.outputHOG())
	{
		// Output the data based on record_root, but do not include record_root in the meta file, as it is also in that directory
		std::string hog_filename = out_name + ".hog";
		metadata_file << "Output HOG:" << hog_filename << std::endl;
		hog_filename = (fs::path(record_root) / hog_filename).string();
		hog_recorder.Open(hog_filename);
	}
		
	// saving the videos	
	if (params.outputTracked())
	{
		if (params.isSequence())
		{
			// Output the data based on record_root, but do not include record_root in the meta file, as it is also in that directory
			this->media_filename = out_name + ".avi";
			metadata_file << "Output video:" << this->media_filename << std::endl;
			this->media_filename = (fs::path(record_root) / this->media_filename).string();
		}
		else
		{
			this->media_filename = out_name + "." + params.imageFormatVisualization();
			metadata_file << "Output image:" << this->media_filename << std::endl;
			this->media_filename = (fs::path(record_root) / this->media_filename).string();
		}
	}

	// Prepare image recording
	if (params.outputAlignedFaces())
	{
		aligned_output_directory = out_name + "_aligned";
		metadata_file << "Output aligned directory:" << this->aligned_output_directory << std::endl;
		this->aligned_output_directory = (fs::path(record_root) / this->aligned_output_directory).string();
		CreateDirectory(aligned_output_directory);		
	}

	this->frame_number = 0;
	this->tracked_writing_thread_started = false;
	this->aligned_writing_thread_started = false;
}

RecorderOpenFace::RecorderOpenFace(const std::string in_filename, const RecorderOpenFaceParameters& parameters, std::vector<std::string>& arguments):video_writer(), params(parameters)
{

	// From the filename, strip out the name without directory and extension
	if (fs::is_directory(in_filename))
	{
		out_name = fs::canonical(in_filename).filename().string();
	}
	else
	{
		out_name = fs::path(in_filename).filename().replace_extension("").string();
	}

	// Consuming the input arguments
	bool* valid = new bool[arguments.size()];

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		valid[i] = true;
	}

	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (arguments[i].compare("-out_dir") == 0)
		{
			record_root = arguments[i + 1];
		}
	}

	// Determine output directory
	bool output_found = false;
	for (size_t i = 0; i < arguments.size(); ++i)
	{
		if (!output_found && arguments[i].compare("-of") == 0)
		{
			record_root = (fs::path(record_root) / fs::path(arguments[i + 1])).remove_filename().string();
			out_name = fs::path(fs::path(arguments[i + 1])).replace_extension("").filename().string();
			valid[i] = false;
			valid[i + 1] = false;
			i++;
			output_found = true;
		}
	}

	// If recording directory not set, record to default location
	if (record_root.empty())
		record_root = default_record_directory;

	for (int i = (int)arguments.size() - 1; i >= 0; --i)
	{
		if (!valid[i])
		{
			arguments.erase(arguments.begin() + i);
		}
	}

	PrepareRecording(in_filename);

}

RecorderOpenFace::RecorderOpenFace(const std::string in_filename, const RecorderOpenFaceParameters& parameters, std::string output_directory):video_writer(), params(parameters)
{
	// From the filename, strip out the name without directory and extension
	if (fs::is_directory(in_filename))
	{
		out_name = fs::canonical(fs::path(in_filename)).filename().string();
	}
	else
	{
		out_name = fs::path(in_filename).filename().replace_extension("").string();
	}

	record_root = output_directory;

	// If recording directory not set, record to default location
	if (record_root.empty())
		record_root = default_record_directory;

	PrepareRecording(in_filename);
}


void RecorderOpenFace::SetObservationFaceAlign(const cv::Mat& aligned_face)
{
	this->aligned_face = aligned_face;
}

void RecorderOpenFace::SetObservationVisualization(const cv::Mat &vis_track)
{
	if (params.outputTracked())
	{
		vis_to_out = vis_track;
	}

}

void RecorderOpenFace::WriteObservation()
{

	// Write out the CSV file (it will always be there, even if not outputting anything more but frame/face numbers)	
	if(!csv_recorder.isOpen())
	{
		// As we are writing out the header, work out some things like number of landmarks, names of AUs etc.
		int num_face_landmarks = landmarks_2D.rows / 2;
		int num_eye_landmarks = (int)eye_landmarks2D.size();
		int num_model_modes = pdm_params_local.rows;

		std::vector<std::string> au_names_class;
		for (auto au : au_occurences)
		{
			au_names_class.push_back(au.first);
		}

		std::sort(au_names_class.begin(), au_names_class.end());

		std::vector<std::string> au_names_reg;
		for (auto au : au_intensities)
		{
			au_names_reg.push_back(au.first);
		}

		std::sort(au_names_reg.begin(), au_names_reg.end());

		metadata_file << "Output csv:" << csv_filename << std::endl;
		metadata_file << "Gaze: " << params.outputGaze() << std::endl;
		metadata_file << "AUs: " << params.outputAUs() << std::endl;
		metadata_file << "Landmarks 2D: " << params.output2DLandmarks() << std::endl;
		metadata_file << "Landmarks 3D: " << params.output3DLandmarks() << std::endl;
		metadata_file << "Pose: " << params.outputPose() << std::endl;
		metadata_file << "Shape parameters: " << params.outputPDMParams() << std::endl;

		csv_filename = (fs::path(record_root) / csv_filename).string();
		csv_recorder.Open(csv_filename, params.isSequence(), params.output2DLandmarks(), params.output3DLandmarks(), params.outputPDMParams(), params.outputPose(),
			params.outputAUs(), params.outputGaze(), num_face_landmarks, num_model_modes, num_eye_landmarks, au_names_class, au_names_reg);
	}

	this->csv_recorder.WriteLine(face_id, frame_number, timestamp, landmark_detection_success, 
		landmark_detection_confidence, landmarks_2D, landmarks_3D, pdm_params_local, pdm_params_global, head_pose,
		gaze_direction0, gaze_direction1, gaze_angle, eye_landmarks2D, eye_landmarks3D, au_intensities, au_occurences);

	if(params.outputHOG())
	{
		this->hog_recorder.Write();
	}

	// Write aligned faces
	if (params.outputAlignedFaces())
	{

		if (!aligned_writing_thread_started)
		{
			aligned_writing_thread_started = true;
			int capacity = (1024 * 1024 * ALIGNED_QUEUE_CAPACITY) / (aligned_face.size().width *aligned_face.size().height * aligned_face.channels()) + 1;
			aligned_face_queue.set_capacity(capacity);

			// Start the alignment output thread			
			aligned_writing_thread = std::thread(&RecorderOpenFace::AlignedImageWritingTask, this);
		}

		char name[100];

		// Filename is based on frame number (TODO stringstream this)
		if(params.isSequence())
			std::sprintf(name, "frame_det_%02d_%06d.", face_id, frame_number);
		else
			std::sprintf(name, "face_det_%06d.", face_id);

		// Construct the output filename
		std::string out_file = (fs::path(aligned_output_directory) / fs::path(std::string(name) + params.imageFormatAligned())).string();

		if(params.outputBadAligned() || landmark_detection_success)
		{
			aligned_face_queue.push(std::pair<std::string, cv::Mat>(out_file, aligned_face));
		}

		// Clear the image
		aligned_face = cv::Mat();

	}

}

void RecorderOpenFace::WriteObservationTracked()
{

	if (params.outputTracked())
	{

		if (!tracked_writing_thread_started)
		{
			tracked_writing_thread_started = true;
			// Set up the queue for video writing based on output size
			int capacity = (1024 * 1024 * TRACKED_QUEUE_CAPACITY) / (vis_to_out.size().width * vis_to_out.size().height * vis_to_out.channels()) + 1;
			vis_to_out_queue.set_capacity(capacity);

			// Initialize the video writer if it has not been opened yet
			if (params.isSequence())
			{
				std::string output_codec = params.outputCodec();
				try
				{
					video_writer.open(media_filename, cv::VideoWriter::fourcc(output_codec[0], output_codec[1], output_codec[2], output_codec[3]), params.outputFps(), vis_to_out.size(), true);

					if (!video_writer.isOpened())
					{
						WARN_STREAM("Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN.");
					}
				}
				catch (cv::Exception e)
				{
					WARN_STREAM("Could not open VideoWriter, OUTPUT FILE WILL NOT BE WRITTEN. Currently using codec " << output_codec << ", try using an other one (-oc option)");
				}
			}

			// Start the video and tracked image writing thread
			video_writing_thread = std::thread(&RecorderOpenFace::VideoWritingTask, this, params.isSequence());

		}

		if (vis_to_out.empty())
		{
			WARN_STREAM("Output tracked video frame is not set");
		}

		if (params.isSequence())
		{
			vis_to_out_queue.push(std::pair<std::string, cv::Mat>("", vis_to_out));
		}
		else
		{
			vis_to_out_queue.push(std::pair<std::string, cv::Mat>(media_filename, vis_to_out));
		}

		// Clear the output
		vis_to_out = cv::Mat();
	}
}

void RecorderOpenFace::SetObservationHOG(bool good_frame, const cv::Mat_<double>& hog_descriptor, int num_cols, int num_rows, int num_channels)
{
	this->hog_recorder.SetObservationHOG(good_frame, hog_descriptor, num_cols, num_rows, num_channels);
}

void RecorderOpenFace::SetObservationTimestamp(double timestamp)
{
	this->timestamp = timestamp;
}

// Required observations for video/image-sequence
void RecorderOpenFace::SetObservationFrameNumber(int frame_number)
{
	this->frame_number = frame_number;
}

// If in multiple face mode, identifying which face was tracked
void RecorderOpenFace::SetObservationFaceID(int face_id)
{
	this->face_id = face_id;
}


void RecorderOpenFace::SetObservationLandmarks(const cv::Mat_<float>& landmarks_2D, const cv::Mat_<float>& landmarks_3D,
	const cv::Vec6f& pdm_params_global, const cv::Mat_<float>& pdm_params_local, double confidence, bool success)
{
	this->landmarks_2D = landmarks_2D;
	this->landmarks_3D = landmarks_3D;
	this->pdm_params_global = pdm_params_global;
	this->pdm_params_local = pdm_params_local;
	this->landmark_detection_confidence = confidence;
	this->landmark_detection_success = success;

}

void RecorderOpenFace::SetObservationPose(const cv::Vec6f& pose)
{
	this->head_pose = pose;
}

void RecorderOpenFace::SetObservationActionUnits(const std::vector<std::pair<std::string, double> >& au_intensities,
	const std::vector<std::pair<std::string, double> >& au_occurences)
{
	this->au_intensities = au_intensities;
	this->au_occurences = au_occurences;
}

void RecorderOpenFace::SetObservationGaze(const cv::Point3f& gaze_direction0, const cv::Point3f& gaze_direction1,
	const cv::Vec2f& gaze_angle, const std::vector<cv::Point2f>& eye_landmarks2D, const std::vector<cv::Point3f>& eye_landmarks3D)
{
	this->gaze_direction0 = gaze_direction0;
	this->gaze_direction1 = gaze_direction1;
	this->gaze_angle = gaze_angle;
	this->eye_landmarks2D = eye_landmarks2D;
	this->eye_landmarks3D = eye_landmarks3D;
}

RecorderOpenFace::~RecorderOpenFace()
{
	this->Close();
}


void RecorderOpenFace::Close()
{
	// Insert terminating frames to the queues
	vis_to_out_queue.push(std::pair<std::string, cv::Mat>("", cv::Mat()));
	aligned_face_queue.push(std::pair<std::string, cv::Mat>("", cv::Mat()));

	// Make sure the recording threads complete
	if (video_writing_thread.joinable())
		video_writing_thread.join();
	if (aligned_writing_thread.joinable())
		aligned_writing_thread.join();

	tracked_writing_thread_started = false;
	aligned_writing_thread_started = false;

	hog_recorder.Close();
	csv_recorder.Close();
	video_writer.release();
	metadata_file.close();
}



