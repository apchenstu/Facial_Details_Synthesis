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

#include "VisualizationUtils.h"
#include "RotationHelpers.h"

namespace Utilities
{

	FpsTracker::FpsTracker()
	{
		// Keep two seconds of history
		history_length = 2;
	}

	void FpsTracker::AddFrame()
	{
		double current_time = cv::getTickCount() / cv::getTickFrequency();
		frame_times.push(current_time);
		DiscardOldFrames();
	}

	double FpsTracker::GetFPS()
	{
		DiscardOldFrames();

		if (frame_times.size() == 0)
			return 0;

		double current_time = cv::getTickCount() / cv::getTickFrequency();

		return ((double)frame_times.size()) / (current_time - frame_times.front());
	}

	void FpsTracker::DiscardOldFrames()
	{
		double current_time = cv::getTickCount() / cv::getTickFrequency();
		// Remove old history
		while (frame_times.size() > 0 && (current_time - frame_times.front()) > history_length)
			frame_times.pop();
	}

	void DrawBox(cv::Mat image, cv::Vec6f pose, cv::Scalar color, int thickness, float fx, float fy, float cx, float cy)
	{
		auto edge_lines = CalculateBox(pose, fx, fy, cx, cy);
		DrawBox(edge_lines, image, color, thickness);
	}

	std::vector<std::pair<cv::Point2f, cv::Point2f>> CalculateBox(cv::Vec6f pose, float fx, float fy, float cx, float cy)
	{
		float boxVerts[] = { -1, 1, -1,
			1, 1, -1,
			1, 1, 1,
			-1, 1, 1,
			1, -1, 1,
			1, -1, -1,
			-1, -1, -1,
			-1, -1, 1 };

		std::vector<std::pair<int, int>> edges;
		edges.push_back(std::pair<int, int>(0, 1));
		edges.push_back(std::pair<int, int>(1, 2));
		edges.push_back(std::pair<int, int>(2, 3));
		edges.push_back(std::pair<int, int>(0, 3));
		edges.push_back(std::pair<int, int>(2, 4));
		edges.push_back(std::pair<int, int>(1, 5));
		edges.push_back(std::pair<int, int>(0, 6));
		edges.push_back(std::pair<int, int>(3, 7));
		edges.push_back(std::pair<int, int>(6, 5));
		edges.push_back(std::pair<int, int>(5, 4));
		edges.push_back(std::pair<int, int>(4, 7));
		edges.push_back(std::pair<int, int>(7, 6));

		// The size of the head is roughly 200mm x 200mm x 200mm
		cv::Mat_<float> box = cv::Mat(8, 3, CV_32F, boxVerts).clone() * 100.0f;

		cv::Matx33f rot = Euler2RotationMatrix(cv::Vec3f(pose[3], pose[4], pose[5]));
		cv::Mat_<float> rotBox;

		// Rotate the box
		rotBox = cv::Mat(rot) * box.t();
		rotBox = rotBox.t();

		// Move the bounding box to head position
		rotBox.col(0) = rotBox.col(0) + pose[0];
		rotBox.col(1) = rotBox.col(1) + pose[1];
		rotBox.col(2) = rotBox.col(2) + pose[2];

		// draw the lines
		cv::Mat_<float> rotBoxProj;
		Project(rotBoxProj, rotBox, fx, fy, cx, cy);

		std::vector<std::pair<cv::Point2f, cv::Point2f>> lines;

		for (size_t i = 0; i < edges.size(); ++i)
		{
			cv::Mat_<float> begin;
			cv::Mat_<float> end;

			rotBoxProj.row(edges[i].first).copyTo(begin);
			rotBoxProj.row(edges[i].second).copyTo(end);

			cv::Point2f p1(begin.at<float>(0), begin.at<float>(1));
			cv::Point2f p2(end.at<float>(0), end.at<float>(1));

			lines.push_back(std::pair<cv::Point2f, cv::Point2f>(p1, p2));

		}

		return lines;
	}

	void DrawBox(const std::vector<std::pair<cv::Point2f, cv::Point2f>>& lines, cv::Mat image, cv::Scalar color, int thickness)
	{
		cv::Rect image_rect(0, 0, image.cols, image.rows);

		for (size_t i = 0; i < lines.size(); ++i)
		{
			cv::Point2f p1 = lines.at(i).first;
			cv::Point2f p2 = lines.at(i).second;
			// Only draw the line if one of the points is inside the image
			if (p1.inside(image_rect) || p2.inside(image_rect))
			{
				cv::line(image, p1, p2, color, thickness, cv::LINE_AA);
			}

		}

	}

	void Visualise_FHOG(const cv::Mat_<double>& descriptor, int num_rows, int num_cols, cv::Mat& visualisation)
	{

		// First convert to dlib format
		dlib::array2d<dlib::matrix<float, 31, 1> > hog(num_rows, num_cols);

		cv::MatConstIterator_<double> descriptor_it = descriptor.begin();
		for (int y = 0; y < num_cols; ++y)
		{
			for (int x = 0; x < num_rows; ++x)
			{
				for (unsigned int o = 0; o < 31; ++o)
				{
					hog[y][x](o) = *descriptor_it++;
				}
			}
		}

		// Draw the FHOG to OpenCV format
		auto fhog_vis = dlib::draw_fhog(hog);
		visualisation = dlib::toMat(fhog_vis).clone();
	}

}