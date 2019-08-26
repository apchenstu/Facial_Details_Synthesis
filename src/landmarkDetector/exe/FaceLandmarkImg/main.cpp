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
// FaceLandmarkImg.cpp : Defines the entry point for the console application for detecting landmarks in images.

// dlib
#include <dlib/image_processing/frontal_face_detector.h>

#include "LandmarkCoreIncludes.h"

#include <FaceAnalyser.h>

#include <ImageCapture.h>
#include <direct.h>


#ifndef CONFIG_DIR
#define CONFIG_DIR "~"
#endif

std::vector<std::string> get_arguments(int argc, char **argv)
{

	std::vector<std::string> arguments;

	for (int i = 0; i < argc; ++i)
	{
		arguments.push_back(std::string(argv[i]));
	}
	return arguments;
}

int main(int argc, char **argv)
{

	//Convert arguments to more convenient vector form
	std::vector<std::string> arguments = get_arguments(argc, argv);

	// no arguments: output usage
	if (arguments.size() == 1)
	{
		std::cout << "For command line arguments see:" << std::endl;
		std::cout << " https://github.com/TadasBaltrusaitis/OpenFace/wiki/Command-line-arguments";
		return 0;
	}

	// Prepare for image reading
	Utilities::ImageCapture image_reader;

	//// The sequence reader chooses what to open based on command line arguments provided
	if (!image_reader.Open(arguments))
	{
		std::cout << "Could not open any images" << std::endl;
		return 1;
	}

	// Load the models if images found
	LandmarkDetector::FaceModelParameters det_parameters(arguments);

	// The modules that are being used for tracking
	std::cout << "Loading the model" << std::endl;
	LandmarkDetector::CLNF face_model(det_parameters.model_location);

	if (!face_model.loaded_successfully)
	{
		std::cout << "ERROR: Could not load the landmark detector" << std::endl;
		return 1;
	}

	std::cout << "Model loaded" << std::endl;

	// Load facial feature extractor and AU analyser (make sure it is static)
	FaceAnalysis::FaceAnalyserParameters face_analysis_params(arguments);
	face_analysis_params.OptimizeForImages();
	FaceAnalysis::FaceAnalyser face_analyser(face_analysis_params);

	// If bounding boxes not provided, use a face detector
	cv::CascadeClassifier classifier(det_parameters.haar_face_detector_location);
	dlib::frontal_face_detector face_detector_hog = dlib::get_frontal_face_detector();
	LandmarkDetector::FaceDetectorMTCNN face_detector_mtcnn(det_parameters.mtcnn_face_detector_location);

	cv::Mat rgb_image;

	rgb_image = image_reader.GetNextImage();
	
	if (!face_model.eye_model)
	{
		std::cout << "WARNING: no eye model found" << std::endl;
	}

	if (face_analyser.GetAUClassNames().size() == 0 && face_analyser.GetAUClassNames().size() == 0)
	{
		std::cout << "WARNING: no Action Unit models found" << std::endl;
	}

	std::cout << "Starting tracking" << std::endl;
	while (!rgb_image.empty())
	{
	
		// Making sure the image is in uchar grayscale (some face detectors use RGB, landmark detector uses grayscale)
		cv::Mat_<uchar> grayscale_image = image_reader.GetGrayFrame();

		// Detect faces in an image
		std::vector<cv::Rect_<float> > face_detections;
		std::vector<float> confidences;
		if (image_reader.has_bounding_boxes)
		{
			face_detections = image_reader.GetBoundingBoxes();
		}
		else
		{

			LandmarkDetector::DetectFacesMTCNN(face_detections, rgb_image, face_detector_mtcnn, confidences);
		}

		char drive[_MAX_DRIVE];
		char dir[_MAX_DIR];
		char fname[_MAX_FNAME];
		char ext[_MAX_EXT];
		std::string name = image_reader.name;
		std::cout << name << std::endl;
		_splitpath(name.c_str(), drive, dir, fname, ext);
		mkdir("./processed/");
		std::ofstream fs("./processed/" + std::string(fname) + ".txt");
		std::ofstream fLandmarks("./processed/" + std::string(fname) + ".pts");
		std::ofstream fbox("./processed/" + std::string(fname) + ".box");

		// Detect landmarks around detected faces
		int face_det = 0;
		// perform landmark detection for every face detected
		for (size_t face = 0; face < face_detections.size(); ++face)
		{

			// if there are multiple detections go through them
			bool success = LandmarkDetector::DetectLandmarksInImage(rgb_image, face_detections[face], face_model, det_parameters, grayscale_image);

			// Estimate head pose and eye gaze				
			cv::Vec6d pose_estimate = LandmarkDetector::GetPose(face_model, image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy);

			

			face_analyser.PredictStaticAUsAndComputeFeatures(rgb_image, face_model.detected_landmarks);

			fs << confidences[face] << " " << pose_estimate[3] << " " << pose_estimate[4] << " " << pose_estimate[5] << " ";

			std::vector<std::pair<std::string, double>> AUsReg = face_analyser.GetCurrentAUsReg();
			for (size_t AU = 0; AU < AUsReg.size(); AU++) {
				fs << AUsReg[AU].second << " ";
			}

			// 3D Pose
			cv::Mat_<float> points = face_model.GetShape(image_reader.fx, image_reader.fy, image_reader.cx, image_reader.cy);
			for (size_t idx = 0; idx < points.cols; idx++) {
				fs << points.at<float>(0, idx) << " " << points.at<float>(1, idx) << " " << points.at<float>(2, idx) << " ";
			}
			fs << std::endl;

			cv::Rect_<float> box = face_model.GetBoundingBox();
			fbox << box.x << " " << box.y << " " << box.width << " " << box.height << std::endl;
			fbox.close();

			//save landmark
			if (0 == face) {
				float scale = 1;
				//float scale = max(255.0/image_reader.image_width, 255.0 / image_reader.image_height);
				fLandmarks << "version: 1\nn_points:  68\n{\n";
				for (size_t idx = 0; idx < 68; idx++) {
					fLandmarks << face_model.detected_landmarks.at<float>(idx, 0)*scale << " ";
					fLandmarks << face_model.detected_landmarks.at<float>(68 + idx, 0)*scale << std::endl;
				}
				fLandmarks << "}\n";
			}

		}
		fs.close(); fLandmarks.close();

		// Grabbing the next frame in the sequence
		rgb_image = image_reader.GetNextImage();

	}

	return 0;
}

