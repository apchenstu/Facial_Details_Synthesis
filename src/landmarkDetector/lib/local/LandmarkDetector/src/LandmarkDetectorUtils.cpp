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

#include <LandmarkDetectorUtils.h>
#include <RotationHelpers.h>

// OpenCV includes
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/calib3d.hpp>

namespace LandmarkDetector
{

	//===========================================================================
	// Fast patch expert response computation (linear model across a ROI) using normalised cross-correlation
	//===========================================================================

	void crossCorr_m(const cv::Mat_<float>& img, cv::Mat_<double>& img_dft, const cv::Mat_<float>& _templ, 
		std::map<int, cv::Mat_<double> >& _templ_dfts, cv::Mat_<float>& corr)
	{
		// Our model will always be under min block size so can ignore this
		//const double blockScale = 4.5;
		//const int minBlockSize = 256;

		int maxDepth = CV_64F;

		cv::Size dftsize;

		dftsize.width = cv::getOptimalDFTSize(corr.cols + _templ.cols - 1);
		dftsize.height = cv::getOptimalDFTSize(corr.rows + _templ.rows - 1);

		// Compute block size
		cv::Size blocksize;
		blocksize.width = dftsize.width - _templ.cols + 1;
		blocksize.width = MIN(blocksize.width, corr.cols);
		blocksize.height = dftsize.height - _templ.rows + 1;
		blocksize.height = MIN(blocksize.height, corr.rows);

		cv::Mat_<double> dftTempl;

		// if this has not been precomputed, precompute it, otherwise use it
		if (_templ_dfts.find(dftsize.width) == _templ_dfts.end())
		{
			dftTempl.create(dftsize.height, dftsize.width);

			cv::Mat_<float> src = _templ;

			cv::Mat_<double> dst(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));

			cv::Mat_<double> dst1(dftTempl, cv::Rect(0, 0, _templ.cols, _templ.rows));

			if (dst1.data != src.data)
				src.convertTo(dst1, dst1.depth());

			if (dst.cols > _templ.cols)
			{
				cv::Mat_<double> part(dst, cv::Range(0, _templ.rows), cv::Range(_templ.cols, dst.cols));
				part.setTo(0);
			}

			// Perform DFT of the template
			dft(dst, dst, 0, _templ.rows);

			_templ_dfts[dftsize.width] = dftTempl;

		}
		else
		{
			// use the precomputed version
			dftTempl = _templ_dfts.find(dftsize.width)->second;
		}

		cv::Size bsz(std::min(blocksize.width, corr.cols), std::min(blocksize.height, corr.rows));
		cv::Mat src;

		cv::Mat cdst(corr, cv::Rect(0, 0, bsz.width, bsz.height));

		cv::Mat_<double> dftImg;

		if (img_dft.empty())
		{
			dftImg.create(dftsize);
			dftImg.setTo(0.0);

			cv::Size dsz(bsz.width + _templ.cols - 1, bsz.height + _templ.rows - 1);

			int x2 = std::min(img.cols, dsz.width);
			int y2 = std::min(img.rows, dsz.height);

			cv::Mat src0(img, cv::Range(0, y2), cv::Range(0, x2));
			cv::Mat dst(dftImg, cv::Rect(0, 0, dsz.width, dsz.height));
			cv::Mat dst1(dftImg, cv::Rect(0, 0, x2, y2));

			src = src0;

			if (dst1.data != src.data)
				src.convertTo(dst1, dst1.depth());

			dft(dftImg, dftImg, 0, dsz.height);
			img_dft = dftImg.clone();
		}

		cv::Mat dftTempl1(dftTempl, cv::Rect(0, 0, dftsize.width, dftsize.height));
		cv::mulSpectrums(img_dft, dftTempl1, dftImg, 0, true);
		cv::dft(dftImg, dftImg, cv::DFT_INVERSE + cv::DFT_SCALE, bsz.height);

		src = dftImg(cv::Rect(0, 0, bsz.width, bsz.height));

		src.convertTo(cdst, CV_32F);

	}

	////////////////////////////////////////////////////////////////////////////////////////////////////////

	void matchTemplate_m(const cv::Mat_<float>& input_img, cv::Mat_<double>& img_dft, cv::Mat& _integral_img, 
		cv::Mat& _integral_img_sq, const cv::Mat_<float>&  templ, std::map<int, cv::Mat_<double> >& templ_dfts, 
		cv::Mat_<float>& result, int method)
	{

		int numType = method == cv::TM_CCORR || method == cv::TM_CCORR_NORMED ? 0 :
			method == cv::TM_CCOEFF || method == cv::TM_CCOEFF_NORMED ? 1 : 2;
		bool isNormed = method == cv::TM_CCORR_NORMED ||
			method == cv::TM_SQDIFF_NORMED ||
			method == cv::TM_CCOEFF_NORMED;

		// Assume result is defined properly
		if (result.empty())
		{
			cv::Size corrSize(input_img.cols - templ.cols + 1, input_img.rows - templ.rows + 1);
			result.create(corrSize);
		}
		LandmarkDetector::crossCorr_m(input_img, img_dft, templ, templ_dfts, result);

		if (method == cv::TM_CCORR)
			return;

		double invArea = 1. / ((double)templ.rows * templ.cols);

		cv::Mat sum, sqsum;
		cv::Scalar templMean, templSdv;
		double *q0 = 0, *q1 = 0, *q2 = 0, *q3 = 0;
		double templNorm = 0, templSum2 = 0;

		if (method == cv::TM_CCOEFF)
		{
			// If it has not been precomputed compute it now
			if (_integral_img.empty())
			{
				integral(input_img, _integral_img, CV_64F);
			}
			sum = _integral_img;

			templMean = cv::mean(templ);
		}
		else
		{
			// If it has not been precomputed compute it now
			if (_integral_img.empty())
			{
				integral(input_img, _integral_img, _integral_img_sq, CV_64F);
			}

			sum = _integral_img;
			sqsum = _integral_img_sq;

			meanStdDev(templ, templMean, templSdv);

			templNorm = templSdv[0] * templSdv[0] + templSdv[1] * templSdv[1] + templSdv[2] * templSdv[2] + templSdv[3] * templSdv[3];

			if (templNorm < DBL_EPSILON && method == cv::TM_CCOEFF_NORMED)
			{
				result.setTo(1.0);
				return;
			}

			templSum2 = templNorm + templMean[0] * templMean[0] + templMean[1] * templMean[1] + templMean[2] * templMean[2] + templMean[3] * templMean[3];

			if (numType != 1)
			{
				templMean = cv::Scalar::all(0);
				templNorm = templSum2;
			}

			templSum2 /= invArea;
			templNorm = std::sqrt(templNorm);
			templNorm /= std::sqrt(invArea); // care of accuracy here

			q0 = (double*)sqsum.data;
			q1 = q0 + templ.cols;
			q2 = (double*)(sqsum.data + templ.rows*sqsum.step);
			q3 = q2 + templ.cols;
		}

		double* p0 = (double*)sum.data;
		double* p1 = p0 + templ.cols;
		double* p2 = (double*)(sum.data + templ.rows*sum.step);
		double* p3 = p2 + templ.cols;

		int sumstep = sum.data ? (int)(sum.step / sizeof(double)) : 0;
		int sqstep = sqsum.data ? (int)(sqsum.step / sizeof(double)) : 0;

		int i, j;

		for (i = 0; i < result.rows; i++)
		{
			float* rrow = result.ptr<float>(i);
			int idx = i * sumstep;
			int idx2 = i * sqstep;

			for (j = 0; j < result.cols; j++, idx += 1, idx2 += 1)
			{
				double num = rrow[j], t;
				double wndMean2 = 0, wndSum2 = 0;

				if (numType == 1)
				{

					t = p0[idx] - p1[idx] - p2[idx] + p3[idx];
					wndMean2 += t*t;
					num -= t*templMean[0];

					wndMean2 *= invArea;
				}

				if (isNormed || numType == 2)
				{

					t = q0[idx2] - q1[idx2] - q2[idx2] + q3[idx2];
					wndSum2 += t;

					if (numType == 2)
					{
						num = wndSum2 - 2 * num + templSum2;
						num = MAX(num, 0.);
					}
				}

				if (isNormed)
				{
					t = std::sqrt(MAX(wndSum2 - wndMean2, 0))*templNorm;
					if (fabs(num) < t)
						num /= t;
					else if (fabs(num) < t*1.125)
						num = num > 0 ? 1 : -1;
					else
						num = method != cv::TM_SQDIFF_NORMED ? 0 : 1;
				}

				rrow[j] = (float)num;
			}
		}
	}

	// Useful utility for grabing a bounding box around a set of 2D landmarks (as a 1D 2n x 1 vector of xs followed by doubles or as an n x 2 vector)
	void ExtractBoundingBox(const cv::Mat_<float>& landmarks, float &min_x, float &max_x, float &min_y, float &max_y)
	{

		if (landmarks.cols == 1)
		{
			int n = landmarks.rows / 2;
			cv::MatConstIterator_<float> landmarks_it = landmarks.begin();

			for (int i = 0; i < n; ++i)
			{
				float val = *landmarks_it++;
				
				if (i == 0 || val < min_x)
					min_x = val;

				if (i == 0 || val > max_x)
					max_x = val;

			}

			for (int i = 0; i < n; ++i)
			{
				float val = *landmarks_it++;

				if (i == 0 || val < min_y)
					min_y = val;

				if (i == 0 || val > max_y)
					max_y = val;

			}
		}
		else
		{
			int n = landmarks.rows;
			for (int i = 0; i < n; ++i)
			{
				float val_x = landmarks.at<float>(i, 0);
				float val_y = landmarks.at<float>(i, 1);

				if (i == 0 || val_x < min_x)
					min_x = val_x;

				if (i == 0 || val_x > max_x)
					max_x = val_x;

				if (i == 0 || val_y < min_y)
					min_y = val_y;

				if (i == 0 || val_y > max_y)
					max_y = val_y;

			}

		}


	}

	// Computing landmarks (to be drawn later possibly)
	std::vector<cv::Point2f> CalculateVisibleLandmarks(const cv::Mat_<float>& shape2D, const cv::Mat_<int>& visibilities)
	{
		int n = shape2D.rows / 2;
		std::vector<cv::Point2f> landmarks;

		for (int i = 0; i < n; ++i)
		{
			if (visibilities.at<int>(i))
			{
				cv::Point2f featurePoint(shape2D.at<float>(i), shape2D.at<float>(i + n));

				landmarks.push_back(featurePoint);
			}
		}

		return landmarks;
	}

	// Computing landmarks (to be drawn later possibly)
	std::vector<cv::Point2f> CalculateAllLandmarks(const cv::Mat_<float>& shape2D)
	{

		int n = 0;
		std::vector<cv::Point2f> landmarks;

		if (shape2D.cols == 2)
		{
			n = shape2D.rows;
		}
		else if (shape2D.cols == 1)
		{
			n = shape2D.rows / 2;
		}

		for (int i = 0; i < n; ++i)
		{
			cv::Point2f featurePoint;
			if (shape2D.cols == 1)
			{
				featurePoint = cv::Point2f(shape2D.at<float>(i), shape2D.at<float>(i + n));
			}
			else
			{
				featurePoint = cv::Point2f(shape2D.at<float>(i, 0), shape2D.at<float>(i, 1));
			}

			landmarks.push_back(featurePoint);
		}

		return landmarks;
	}

	// Computing landmarks (to be drawn later possibly)
	std::vector<cv::Point2f> CalculateAllLandmarks(const CLNF& clnf_model)
	{
		return CalculateAllLandmarks(clnf_model.detected_landmarks);
	}

	// Computing landmarks (to be drawn later possibly)
	std::vector<cv::Point2f> CalculateVisibleLandmarks(const CLNF& clnf_model)
	{
		// If the detection was not successful no landmarks are visible
		if (clnf_model.detection_success)
		{
			int idx = clnf_model.patch_experts.GetViewIdx(clnf_model.params_global, 0);
			// Because we only draw visible points, need to find which points patch experts consider visible at a certain orientation
			return CalculateVisibleLandmarks(clnf_model.detected_landmarks, clnf_model.patch_experts.visibilities[0][idx]);
		}
		else
		{
			return std::vector<cv::Point2f>();
		}
	}

	// Computing eye landmarks (to be drawn later or in different interfaces)
	std::vector<cv::Point2f> CalculateVisibleEyeLandmarks(const CLNF& clnf_model)
	{

		std::vector<cv::Point2f> to_return;
		// If the model has hierarchical updates draw those too
		for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
		{

			if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
				clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
			{

				auto lmks = CalculateVisibleLandmarks(clnf_model.hierarchical_models[i]);
				for (auto lmk : lmks)
				{
					to_return.push_back(lmk);
				}
			}
		}
		return to_return;
	}
	// Computing the 3D eye landmarks
	std::vector<cv::Point3f> Calculate3DEyeLandmarks(const CLNF& clnf_model, float fx, float fy, float cx, float cy)
	{

		std::vector<cv::Point3f> to_return;

		for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
		{

			if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
				clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
			{

				auto lmks = clnf_model.hierarchical_models[i].GetShape(fx, fy, cx, cy);

				int num_landmarks = lmks.cols;

				for (int lmk = 0; lmk < num_landmarks; ++lmk)
				{
					cv::Point3f curr_lmk(lmks.at<float>(0, lmk), lmks.at<float>(1, lmk), lmks.at<float>(2, lmk));
					to_return.push_back(curr_lmk);
				}
			}
		}
		return to_return;
	}

	// Computing eye landmarks (to be drawn later or in different interfaces)
	std::vector<cv::Point2f> CalculateAllEyeLandmarks(const CLNF& clnf_model)
	{

		std::vector<cv::Point2f> to_return;
		// If the model has hierarchical updates draw those too
		for (size_t i = 0; i < clnf_model.hierarchical_models.size(); ++i)
		{

			if (clnf_model.hierarchical_model_names[i].compare("left_eye_28") == 0 ||
				clnf_model.hierarchical_model_names[i].compare("right_eye_28") == 0)
			{

				auto lmks = CalculateAllLandmarks(clnf_model.hierarchical_models[i]);
				for (auto lmk : lmks)
				{
					to_return.push_back(lmk);
				}
			}
		}
		return to_return;
	}

	//===========================================================================

	//============================================================================
	// Face detection helpers
	//============================================================================
	bool DetectFaces(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, float min_width, cv::Rect_<float> roi)
	{
		cv::CascadeClassifier classifier("./classifiers/haarcascade_frontalface_alt.xml");
		if (classifier.empty())
		{
			std::cout << "Couldn't load the Haar cascade classifier" << std::endl;
			return false;
		}
		else
		{
			return DetectFaces(o_regions, intensity, classifier, min_width, roi);
		}

	}

	bool DetectFaces(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, cv::CascadeClassifier& classifier, float min_width, cv::Rect_<float> roi)
	{

		std::vector<cv::Rect> face_detections;
		if (min_width == -1)
		{
			classifier.detectMultiScale(intensity, face_detections, 1.2, 2, 0, cv::Size(50, 50));
		}
		else
		{
			classifier.detectMultiScale(intensity, face_detections, 1.2, 2, 0, cv::Size(min_width, min_width));
		}

		// Convert from int bounding box do a double one with corrections
		for (size_t face = 0; face < face_detections.size(); ++face)
		{
			// OpenCV is overgenerous with face size and y location is off
			// CLNF detector expects the bounding box to encompass from eyebrow to chin in y, and from cheeck outline to cheeck outline in x, so we need to compensate

			// The scalings were learned using the Face Detections on LFPW, Helen, AFW and iBUG datasets, using ground truth and detections from openCV
			cv::Rect_<float> region;
			// Correct for scale
			region.width = face_detections[face].width * 0.8924f;
			region.height = face_detections[face].height * 0.8676f;

			// Move the face slightly to the right (as the width was made smaller)
			region.x = face_detections[face].x + 0.0578f * face_detections[face].width;
			// Shift face down as OpenCV Haar Cascade detects the forehead as well, and we're not interested
			region.y = face_detections[face].y + face_detections[face].height * 0.2166f;

			if (min_width != -1)
			{
				if (region.width < min_width || region.x < ((float)intensity.cols) * roi.x || region.y < ((float)intensity.cols) * roi.y || region.x + region.width >((float)intensity.cols) * (roi.x + roi.width) || region.y + region.height >((float)intensity.rows) * (roi.y + roi.height))
					continue;
			}


			o_regions.push_back(region);
		}
		return o_regions.size() > 0;
	}

	bool DetectSingleFace(cv::Rect_<float>& o_region, const cv::Mat_<uchar>& intensity_image, cv::CascadeClassifier& classifier, cv::Point preference, float min_width, cv::Rect_<float> roi)
	{
		// The tracker can return multiple faces
		std::vector<cv::Rect_<float> > face_detections;

		bool detect_success = LandmarkDetector::DetectFaces(face_detections, intensity_image, classifier, min_width, roi);

		if (detect_success)
		{

			bool use_preferred = (preference.x != -1) && (preference.y != -1);

			if (face_detections.size() > 1)
			{
				// keep the closest one if preference point not set
				float best = -1;
				int bestIndex = -1;
				for (size_t i = 0; i < face_detections.size(); ++i)
				{
					float dist;
					bool better;

					if (use_preferred)
					{
						dist = sqrt((preference.x) * (face_detections[i].width / 2 + face_detections[i].x) +
							(preference.y) * (face_detections[i].height / 2 + face_detections[i].y));
						better = dist < best;
					}
					else
					{
						dist = face_detections[i].width;
						better = face_detections[i].width > best;
					}

					// Pick a closest face to preffered point or the biggest face
					if (i == 0 || better)
					{
						bestIndex = i;
						best = dist;
					}
				}

				o_region = face_detections[bestIndex];

			}
			else
			{
				o_region = face_detections[0];
			}

		}
		else
		{
			// if not detected
			o_region = cv::Rect_<float>(0, 0, 0, 0);
		}
		return detect_success;
	}

	bool DetectFacesHOG(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, 
		std::vector<float>& confidences, float min_width, cv::Rect_<float> roi)
	{
		dlib::frontal_face_detector detector = dlib::get_frontal_face_detector();

		return DetectFacesHOG(o_regions, intensity, detector, confidences, min_width, roi);

	}

	bool DetectFacesHOG(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat_<uchar>& intensity, 
		dlib::frontal_face_detector& detector, std::vector<float>& o_confidences, float min_width, cv::Rect_<float> roi)
	{
		if (detector.num_detectors() == 0)
		{
			detector = dlib::get_frontal_face_detector();
		}

		cv::Mat_<uchar> upsampled_intensity;

		float scaling = 1.3f;

		cv::resize(intensity, upsampled_intensity, cv::Size((int)(intensity.cols * scaling), (int)(intensity.rows * scaling)));

		dlib::cv_image<uchar> cv_grayscale(upsampled_intensity);

		std::vector<dlib::full_detection> face_detections;
		detector(cv_grayscale, face_detections, -0.2);

		// Convert from int bounding box do a double one with corrections
		for (size_t face = 0; face < face_detections.size(); ++face)
		{
			// CLNF expects the bounding box to encompass from eyebrow to chin in y, and from cheeck outline to cheeck outline in x, so we need to compensate

			cv::Rect_<float> region;
			// Move the face slightly to the right (as the width was made smaller)
			region.x = (face_detections[face].rect.get_rect().tl_corner().x() + 0.0389f * face_detections[face].rect.get_rect().width()) / scaling;
			// Shift face down as OpenCV Haar Cascade detects the forehead as well, and we're not interested
			region.y = (face_detections[face].rect.get_rect().tl_corner().y() + 0.1278f * face_detections[face].rect.get_rect().height()) / scaling;

			// Correct for scale
			region.width = (face_detections[face].rect.get_rect().width() * 0.9611) / scaling;
			region.height = (face_detections[face].rect.get_rect().height() * 0.9388) / scaling;

			// The scalings were learned using the Face Detections on LFPW and Helen using ground truth and detections from the HOG detector
			if (min_width != -1)
			{
				if (region.width < min_width || region.x < ((float)intensity.cols) * roi.x || region.y < ((float)intensity.cols) * roi.y ||
					region.x + region.width >((float)intensity.cols) * (roi.x + roi.width) || region.y + region.height >((float)intensity.rows) * (roi.y + roi.height))
					continue;
			}


			o_regions.push_back(region);
			o_confidences.push_back(face_detections[face].detection_confidence);


		}
		return o_regions.size() > 0;
	}

	bool DetectSingleFaceHOG(cv::Rect_<float>& o_region, const cv::Mat_<uchar>& intensity_img, dlib::frontal_face_detector& detector, float& confidence, cv::Point preference, float min_width, cv::Rect_<float> roi)
	{

		if (detector.num_detectors() == 0)
		{
			detector = dlib::get_frontal_face_detector();
		}

		// The tracker can return multiple faces
		std::vector<cv::Rect_<float> > face_detections;
		std::vector<float> confidences;
		bool detect_success = LandmarkDetector::DetectFacesHOG(face_detections, intensity_img, detector, confidences, min_width, roi);

		// In case of multiple faces pick the biggest one
		bool use_size = true;

		if (detect_success)
		{

			bool use_preferred = (preference.x != -1) && (preference.y != -1);

			// keep the most confident one or the one closest to preference point if set
			float best_so_far;
			if (use_preferred)
			{
				best_so_far = sqrt((preference.x - (face_detections[0].width / 2 + face_detections[0].x)) * (preference.x - (face_detections[0].width / 2 + face_detections[0].x)) +
					(preference.y - (face_detections[0].height / 2 + face_detections[0].y)) * (preference.y - (face_detections[0].height / 2 + face_detections[0].y)));
			}
			else if (use_size)
			{
				best_so_far = (face_detections[0].width + face_detections[0].height) / 2.0;
			}
			else
			{
				best_so_far = confidences[0];
			}
			int bestIndex = 0;

			for (size_t i = 1; i < face_detections.size(); ++i)
			{

				float dist;
				bool better;

				if (use_preferred)
				{
					dist = sqrt((preference.x - (face_detections[i].width / 2 + face_detections[i].x)) * (preference.x - (face_detections[i].width / 2 + face_detections[i].x)) +
						(preference.y - (face_detections[i].height / 2 + face_detections[i].y)) * (preference.y - (face_detections[i].height / 2 + face_detections[i].y)));

					better = dist < best_so_far;
				}
				else if (use_size)
				{
					dist = (face_detections[i].width + face_detections[i].height) / 2.0;
					better = dist > best_so_far;
				}
				else
				{
					dist = confidences[i];
					better = dist > best_so_far;
				}

				// Pick a closest face
				if (better)
				{
					best_so_far = dist;
					bestIndex = i;
				}
			}

			o_region = face_detections[bestIndex];
			confidence = confidences[bestIndex];
		}
		else
		{
			// if not detected
			o_region = cv::Rect_<float>(0, 0, 0, 0);
			// A completely unreliable detection (shouldn't really matter what is returned here)
			confidence = -2;
		}
		return detect_success;
	}

bool DetectFacesMTCNN(std::vector<cv::Rect_<float> >& o_regions, const cv::Mat& image, LandmarkDetector::FaceDetectorMTCNN& detector, 
	std::vector<float>& o_confidences)
{
	detector.DetectFaces(o_regions, image, o_confidences);

	return o_regions.size() > 0;
}

bool DetectSingleFaceMTCNN(cv::Rect_<float>& o_region, const cv::Mat& image, LandmarkDetector::FaceDetectorMTCNN& detector, 
	float& confidence, cv::Point preference)
{
	// The tracker can return multiple faces
	std::vector<cv::Rect_<float> > face_detections;
	std::vector<float> confidences;

	detector.DetectFaces(face_detections, image, confidences);

	bool detect_success = face_detections.size() > 0;
	if (detect_success)
	{

		bool use_preferred = (preference.x != -1) && (preference.y != -1);

		// keep the most confident one or the one closest to preference point if set
		float best_so_far;
		if (use_preferred)
		{
			best_so_far = sqrt((preference.x - (face_detections[0].width / 2 + face_detections[0].x)) * (preference.x - (face_detections[0].width / 2 + face_detections[0].x)) +
				(preference.y - (face_detections[0].height / 2 + face_detections[0].y)) * (preference.y - (face_detections[0].height / 2 + face_detections[0].y)));
		}
		else
		{
			best_so_far = face_detections[0].width;
		}
		int bestIndex = 0;

		for (size_t i = 1; i < face_detections.size(); ++i)
		{

			float dist;
			bool better;

			if (use_preferred)
			{
				dist = sqrt((preference.x - (face_detections[i].width / 2 + face_detections[i].x)) * (preference.x - (face_detections[i].width / 2 + face_detections[i].x)) +
					(preference.y - (face_detections[i].height / 2 + face_detections[i].y)) * (preference.y - (face_detections[i].height / 2 + face_detections[i].y)));
				better = dist < best_so_far;
			}
			else
			{
				dist = face_detections[i].width;
				better = dist > best_so_far;
			}

			// Pick a closest face
			if (better)
			{
				best_so_far = dist;
				bestIndex = i;
			}
		}

		o_region = face_detections[bestIndex];
		confidence = confidences[bestIndex];
	}
	else
	{
		// if not detected
		o_region = cv::Rect_<float>(0, 0, 0, 0);
		// A completely unreliable detection (shouldn't really matter what is returned here)
		confidence = -2;
	}
	return detect_success;
}


//============================================================================
// Matrix reading functionality
//============================================================================

// Reading in a matrix from a stream
void ReadMat(std::ifstream& stream, cv::Mat &output_mat)
{
	// Read in the number of rows, columns and the data type
	int row, col, type;

	stream >> row >> col >> type;

	output_mat = cv::Mat(row, col, type);

	switch (output_mat.type())
	{
	case CV_64FC1:
	{
		cv::MatIterator_<double> begin_it = output_mat.begin<double>();
		cv::MatIterator_<double> end_it = output_mat.end<double>();

		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_32FC1:
	{
		cv::MatIterator_<float> begin_it = output_mat.begin<float>();
		cv::MatIterator_<float> end_it = output_mat.end<float>();

		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_32SC1:
	{
		cv::MatIterator_<int> begin_it = output_mat.begin<int>();
		cv::MatIterator_<int> end_it = output_mat.end<int>();
		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	case CV_8UC1:
	{
		cv::MatIterator_<uchar> begin_it = output_mat.begin<uchar>();
		cv::MatIterator_<uchar> end_it = output_mat.end<uchar>();
		while (begin_it != end_it)
		{
			stream >> *begin_it++;
		}
	}
	break;
	default:
		printf("ERROR(%s,%d) : Unsupported Matrix type %d!\n", __FILE__, __LINE__, output_mat.type()); abort();


	}
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

// Skipping lines that start with # (together with empty lines)
void SkipComments(std::ifstream& stream)
{
	while (stream.peek() == '#' || stream.peek() == '\n' || stream.peek() == ' ' || stream.peek() == '\r')
	{
		std::string skipped;
		std::getline(stream, skipped);
	}
}

}
