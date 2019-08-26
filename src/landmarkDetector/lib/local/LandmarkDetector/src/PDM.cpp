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

#include <PDM.h>
#include <RotationHelpers.h>

// OpenCV include
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc.hpp>

// Math includes
#define _USE_MATH_DEFINES
#include <cmath>

#ifndef M_PI
	#define M_PI 3.14159265358979323846
#endif

#include <LandmarkDetectorUtils.h>

using namespace LandmarkDetector;
//===========================================================================

//=============================================================================
// Orthonormalising the 3x3 rotation matrix
void PDM::Orthonormalise(cv::Matx33f &R)
{

	cv::SVD svd(R, cv::SVD::MODIFY_A);
  
	// get the orthogonal matrix from the initial rotation matrix
	cv::Mat_ <float> X = svd.u*svd.vt;
  
	// This makes sure that the handedness is preserved and no reflection happened
	// by making sure the determinant is 1 and not -1
	cv::Mat_<float> W = cv::Mat_<float>::eye(3,3);
	float d = determinant(X);
	W(2,2) = determinant(X);
	cv::Mat Rt = svd.u*W*svd.vt;

	Rt.copyTo(R);

}

// A copy constructor
PDM::PDM(const PDM& other) {

	// Make sure the matrices are allocated properly
	this->mean_shape = other.mean_shape.clone();
	this->princ_comp = other.princ_comp.clone();
	this->eigen_values = other.eigen_values.clone();
}

//===========================================================================
// Clamping the parameter values to be within 3 standard deviations
void PDM::Clamp(cv::Mat_<float>& local_params, cv::Vec6f& params_global, const FaceModelParameters& parameters)
{
	float n_sigmas = 3;
	cv::MatConstIterator_<float> e_it  = this->eigen_values.begin();
	cv::MatIterator_<float> p_it =  local_params.begin();

	float v;

	// go over all parameters
	for(; p_it != local_params.end(); ++p_it, ++e_it)
	{
		// Work out the maximum value
		v = n_sigmas*sqrt(*e_it);

		// if the values is too extreme clamp it
		if(fabs(*p_it) > v)
		{
			// Dealing with positive and negative cases
			if(*p_it > 0.0)
			{
				*p_it=v;
			}
			else
			{
				*p_it=-v;
			}
		}
	}
	
	// do not let the pose get out of hand
	//if(parameters.limit_pose)
	//{
	//	if(params_global[1] > M_PI / 2)
	//		params_global[1] = M_PI/2;
	//	if(params_global[1] < -M_PI / 2)
	//		params_global[1] = -M_PI/2;
	//	if(params_global[2] > M_PI / 2)
	//		params_global[2] = M_PI/2;
	//	if(params_global[2] < -M_PI / 2)
	//		params_global[2] = -M_PI/2;
	//	if(params_global[3] > M_PI / 2)
	//		params_global[3] = M_PI/2;
	//	if(params_global[3] < -M_PI / 2)
	//		params_global[3] = -M_PI/2;
	//}
	

}
//===========================================================================
// Compute the 3D representation of shape (in object space) using the local parameters
void PDM::CalcShape3D(cv::Mat_<float>& out_shape, const cv::Mat_<float>& p_local) const
{
	out_shape = mean_shape.clone();

	// Perform matrix vector multiplication in OpenBLAS (fortran call)
	float alpha1 = 1.0;
	float beta1 = 1.0;
	int p_local_cols = p_local.cols;
	int princ_comp_rows = princ_comp.rows;
	int princ_comp_cols = princ_comp.cols;
	char N[2]; N[0] = 'N';
	sgemm_(N, N, &p_local_cols, &princ_comp_rows, &princ_comp_cols, &alpha1, (float*)p_local.data, &p_local_cols, (float*)princ_comp.data, &princ_comp_cols, &beta1, (float*)out_shape.data, &p_local_cols);

	// Above is a fast (but ugly) version of 
	// out_shape = mean_shape + princ_comp * p_local;	 

}

//===========================================================================
// Get the 2D shape (in image space) from global and local parameters
void PDM::CalcShape2D(cv::Mat_<float>& out_shape, const cv::Mat_<float>& params_local, const cv::Vec6f& params_global) const
{

	int n = this->NumberOfPoints();

	float s = params_global[0]; // scaling factor
	float tx = params_global[4]; // x offset
	float ty = params_global[5]; // y offset

	// get the rotation matrix from the euler angles
	cv::Vec3f euler(params_global[1], params_global[2], params_global[3]);
	cv::Matx33f currRot = Utilities::Euler2RotationMatrix(euler);
	
	// get the 3D shape of the object
	cv::Mat_<float> Shape_3D;
	this->CalcShape3D(Shape_3D, params_local);

	// create the 2D shape matrix (if it has not been defined yet)
	if((out_shape.rows != 2 * mean_shape.rows / 3) || (out_shape.cols != 1))
	{
		out_shape.create(2*n,1);
	}
	// for every vertex
	for(int i = 0; i < n; i++)
	{
		// Transform this using the weak-perspective mapping to 2D from 3D
		out_shape.at<float>(i  ,0) = s * ( currRot(0,0) * Shape_3D.at<float>(i, 0) + currRot(0,1) * Shape_3D.at<float>(i+n  ,0) + currRot(0,2) * Shape_3D.at<float>(i+n*2,0) ) + tx;
		out_shape.at<float>(i+n,0) = s * ( currRot(1,0) * Shape_3D.at<float>(i, 0) + currRot(1,1) * Shape_3D.at<float>(i+n  ,0) + currRot(1,2) * Shape_3D.at<float>(i+n*2,0) ) + ty;
	}
}

//===========================================================================
// provided the bounding box of a face and the local parameters (with optional rotation), generates the global parameters that can generate the face with the provided bounding box
// This all assumes that the bounding box describes face from left outline to right outline of the face and chin to eyebrows
void PDM::CalcParams(cv::Vec6f& out_params_global, const cv::Rect_<float>& bounding_box, const cv::Mat_<float>& params_local, const cv::Vec3f rotation)
{

	// get the shape instance based on local params
	cv::Mat_<float> current_shape(mean_shape.size());

	CalcShape3D(current_shape, params_local);

	// rotate the shape
	cv::Matx33f rotation_matrix = Utilities::Euler2RotationMatrix(rotation);

	cv::Mat_<float> reshaped = current_shape.reshape(1, 3);

	cv::Mat rotated_shape = (cv::Mat(rotation_matrix) * reshaped);

	// Get the width of expected shape
	double min_x;
	double max_x;
	cv::minMaxLoc(rotated_shape.row(0), &min_x, &max_x);	

	double min_y;
	double max_y;
	cv::minMaxLoc(rotated_shape.row(1), &min_y, &max_y);

	float width = (float) abs(min_x - max_x);
	float height = (float)abs(min_y - max_y);

	float scaling = ((bounding_box.width / width) + (bounding_box.height / height)) / 2.0f;

	// The estimate of face center also needs some correction
	float tx = bounding_box.x + bounding_box.width / 2;
	float ty = bounding_box.y + bounding_box.height / 2;

	// Correct it so that the bounding box is just around the minimum and maximum point in the initialised face	
	tx = tx - scaling * (min_x + max_x)/2.0f;
    ty = ty - scaling * (min_y + max_y)/2.0f;

	out_params_global = cv::Vec6f(scaling, rotation[0], rotation[1], rotation[2], tx, ty);
}

//===========================================================================
// provided the model parameters, compute the bounding box of a face
// The bounding box describes face from left outline to right outline of the face and chin to eyebrows
void PDM::CalcBoundingBox(cv::Rect_<float>& out_bounding_box, const cv::Vec6f& params_global, const cv::Mat_<float>& params_local)
{
	
	// get the shape instance based on local params
	cv::Mat_<float> current_shape;
	CalcShape2D(current_shape, params_local, params_global);
	
	// Get the width of expected shape
	float min_x, max_x, min_y, max_y;
	ExtractBoundingBox(current_shape, min_x, max_x, min_y, max_y);

	float width = abs(min_x - max_x);
	float height = abs(min_y - max_y);

	out_bounding_box = cv::Rect_<float>(min_x, min_y, width, height);
}

//===========================================================================
// Calculate the PDM's Jacobian over rigid parameters (rotation, translation and scaling), the additional input W represents trust for each of the landmarks and is part of Non-Uniform RLMS 
void PDM::ComputeRigidJacobian(const cv::Mat_<float>& p_local, const cv::Vec6f& params_global, cv::Mat_<float> &Jacob, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w)
{
  	
	// number of verts
	int n = this->NumberOfPoints();
  
	Jacob.create(n * 2, 6);

	float X,Y,Z;

	float s = params_global[0];
  	
	cv::Mat_<float> shape_3D;
	this->CalcShape3D(shape_3D, p_local);
		
	 // Get the rotation matrix
	cv::Vec3f euler(params_global[1], params_global[2], params_global[3]);
	cv::Matx33f currRot = Utilities::Euler2RotationMatrix(euler);
	
	float r11 = currRot(0,0);
	float r12 = currRot(0,1);
	float r13 = currRot(0,2);
	float r21 = currRot(1,0);
	float r22 = currRot(1,1);
	float r23 = currRot(1,2);
	float r31 = currRot(2,0);
	float r32 = currRot(2,1);
	float r33 = currRot(2,2);

	cv::MatIterator_<float> Jx = Jacob.begin();
	cv::MatIterator_<float> Jy = Jx + n * 6;

	for(int i = 0; i < n; i++)
	{
    
		X = shape_3D.at<float>(i, 0);
		Y = shape_3D.at<float>(i + n, 0);
		Z = shape_3D.at<float>(i + n * 2, 0);
		
		// The rigid jacobian from the axis angle rotation matrix approximation using small angle assumption (R * R')
		// where R' = [1, -wz, wy
		//             wz, 1, -wx
		//             -wy, wx, 1]
		// And this is derived using the small angle assumption on the axis angle rotation matrix parametrisation

		// scaling term
		*Jx++ =  (X  * r11 + Y * r12 + Z * r13);
		*Jy++ =  (X  * r21 + Y * r22 + Z * r23);
		
		// rotation terms
		*Jx++ = (s * (Y * r13 - Z * r12) );
		*Jy++ = (s * (Y * r23 - Z * r22) );
		*Jx++ = (-s * (X * r13 - Z * r11));
		*Jy++ = (-s * (X * r23 - Z * r21));
		*Jx++ = (s * (X * r12 - Y * r11) );
		*Jy++ = (s * (X * r22 - Y * r21) );
		
		// translation terms
		*Jx++ = 1.0f;
		*Jy++ = 0.0f;
		*Jx++ = 0.0f;
		*Jy++ = 1.0f;

	}

	cv::Mat Jacob_w = cv::Mat::zeros(Jacob.rows, Jacob.cols, Jacob.type());
	
	Jx =  Jacob.begin();
	Jy =  Jx + n*6;

	cv::MatIterator_<float> Jx_w =  Jacob_w.begin<float>();
	cv::MatIterator_<float> Jy_w =  Jx_w + n*6;

	// Iterate over all Jacobian values and multiply them by the weight in diagonal of W
	for(int i = 0; i < n; i++)
	{
		float w_x = W.at<float>(i, i);
		float w_y = W.at<float>(i+n, i+n);

		for(int j = 0; j < Jacob.cols; ++j)
		{
			*Jx_w++ = *Jx++ * w_x;
			*Jy_w++ = *Jy++ * w_y;
		}		
	}

	Jacob_t_w = Jacob_w.t();
}

//===========================================================================
// Calculate the PDM's Jacobian over all parameters (rigid and non-rigid), the additional input W represents trust for each of the landmarks and is part of Non-Uniform RLMS
void PDM::ComputeJacobian(const cv::Mat_<float>& params_local, const cv::Vec6f& params_global, cv::Mat_<float> &Jacobian, const cv::Mat_<float> W, cv::Mat_<float> &Jacob_t_w)
{ 
	
	// number of vertices
	int n = this->NumberOfPoints();
		
	// number of non-rigid parameters
	int m = this->NumberOfModes();

	Jacobian.create(n * 2, 6 + m);
	
	float X,Y,Z;
	
	float s = params_global[0];
  	
	cv::Mat_<float> shape_3D;
	this->CalcShape3D(shape_3D, params_local);
	
	cv::Vec3f euler(params_global[1], params_global[2], params_global[3]);
	cv::Matx33f currRot = Utilities::Euler2RotationMatrix(euler);
	
	float r11 = currRot(0, 0);
	float r12 = currRot(0, 1);
	float r13 = currRot(0, 2);
	float r21 = currRot(1, 0);
	float r22 = currRot(1, 1);
	float r23 = currRot(1, 2);
	float r31 = currRot(2, 0);
	float r32 = currRot(2, 1);
	float r33 = currRot(2, 2);

	cv::MatIterator_<float> Jx =  Jacobian.begin();
	cv::MatIterator_<float> Jy =  Jx + n * (6 + m);
	cv::MatConstIterator_<float> Vx =  this->princ_comp.begin();
	cv::MatConstIterator_<float> Vy =  Vx + n*m;
	cv::MatConstIterator_<float> Vz =  Vy + n*m;

	for(int i = 0; i < n; i++)
	{
    
		X = shape_3D.at<float>(i, 0);
		Y = shape_3D.at<float>(i + n, 0);
		Z = shape_3D.at<float>(i + n * 2, 0);
    
		// The rigid jacobian from the axis angle rotation matrix approximation using small angle assumption (R * R')
		// where R' = [1, -wz, wy
		//             wz, 1, -wx
		//             -wy, wx, 1]
		// And this is derived using the small angle assumption on the axis angle rotation matrix parametrisation

		// scaling term
		*Jx++ = (X  * r11 + Y * r12 + Z * r13);
		*Jy++ = (X  * r21 + Y * r22 + Z * r23);
		
		// rotation terms
		*Jx++ = (s * (Y * r13 - Z * r12) );
		*Jy++ = (s * (Y * r23 - Z * r22) );
		*Jx++ = (-s * (X * r13 - Z * r11));
		*Jy++ = (-s * (X * r23 - Z * r21));
		*Jx++ = (s * (X * r12 - Y * r11) );
		*Jy++ = (s * (X * r22 - Y * r21) );
		
		// translation terms
		*Jx++ = 1.0f;
		*Jy++ = 0.0f;
		*Jx++ = 0.0f;
		*Jy++ = 1.0f;

		for(int j = 0; j < m; j++,++Vx,++Vy,++Vz)
		{
			// How much the change of the non-rigid parameters (when object is rotated) affect 2D motion
			*Jx++ = ( s*(r11*(*Vx) + r12*(*Vy) + r13*(*Vz)) );
			*Jy++ = ( s*(r21*(*Vx) + r22*(*Vy) + r23*(*Vz)) );
		}
	}	

	// Adding the weights here	
	if(cv::trace(W)[0] != W.rows) 
	{
		cv::Mat Jacob_w = Jacobian.clone();
		Jx =  Jacobian.begin();
		Jy =  Jx + n*(6+m);

		cv::MatIterator_<float> Jx_w =  Jacob_w.begin<float>();
		cv::MatIterator_<float> Jy_w =  Jx_w + n*(6+m);

		// Iterate over all Jacobian values and multiply them by the weight in diagonal of W
		for(int i = 0; i < n; i++)
		{
			float w_x = W.at<float>(i, i);
			float w_y = W.at<float>(i+n, i+n);

			for(int j = 0; j < Jacobian.cols; ++j)
			{
				*Jx_w++ = *Jx++ * w_x;
				*Jy_w++ = *Jy++ * w_y;
			}
		}
		Jacob_t_w = Jacob_w.t();
	}
	else
	{
		Jacob_t_w = Jacobian.t();
	}
}

//===========================================================================
// Updating the parameters (more details in my thesis)
void PDM::UpdateModelParameters(const cv::Mat_<float>& delta_p, cv::Mat_<float>& params_local, cv::Vec6f& params_global)
{

	// The scaling and translation parameters can be just added
	params_global[0] += delta_p.at<float>(0,0);
	params_global[4] += delta_p.at<float>(4,0);
	params_global[5] += delta_p.at<float>(5,0);

	// get the original rotation matrix	
	cv::Vec3f eulerGlobal(params_global[1], params_global[2], params_global[3]);
	
	cv::Matx33f R1 = Utilities::Euler2RotationMatrix(eulerGlobal);

	// construct R' = [1, -wz, wy
	//               wz, 1, -wx
	//               -wy, wx, 1]
	cv::Matx33f R2 = cv::Matx33f::eye();

	R2(1,2) = -1.0*(R2(2,1) = delta_p.at<float>(1,0));
	R2(2,0) = -1.0*(R2(0,2) = delta_p.at<float>(2,0));
	R2(0,1) = -1.0*(R2(1,0) = delta_p.at<float>(3,0));
	
	// Make sure it's orthonormal
	Orthonormalise(R2);

	// Combine rotations
	cv::Matx33f R3 = R1 *R2;

	// Extract euler angle (through axis angle first to make sure it's legal)
	cv::Vec3f axis_angle = Utilities::RotationMatrix2AxisAngle(R3);

	cv::Vec3f euler = Utilities::AxisAngle2Euler(axis_angle);

	// Temporary fix to numerical instability
	if (std::isnan(euler[0]) || std::isnan(euler[1]) || std::isnan(euler[2]))
	{
		euler[0] = 0;
		euler[1] = 0;
		euler[2] = 0;

	}

	params_global[1] = euler[0];
	params_global[2] = euler[1];
	params_global[3] = euler[2];

	// Local parameter update, just simple addition
	if(delta_p.rows > 6)
	{
		params_local = params_local + delta_p(cv::Rect(0,6,1, this->NumberOfModes()));
	}

}

void PDM::CalcParams(cv::Vec6f& out_params_global, cv::Mat_<float>& out_params_local, const cv::Mat_<float> & landmark_locations, const cv::Vec3f rotation)
{
		
	int m = this->NumberOfModes();
	int n = this->NumberOfPoints();

	cv::Mat_<int> visi_ind_2D(n * 2, 1, 1);
	cv::Mat_<int> visi_ind_3D(3 * n , 1, 1);

	int visi_count = n;

	for(int i = 0; i < n; ++i)
	{
		// If the landmark is invisible indicate this
		if(landmark_locations.at<float>(i) == 0)
		{
			visi_ind_2D.at<int>(i) = 0;
			visi_ind_2D.at<int>(i+n) = 0;
			visi_ind_3D.at<int>(i) = 0;
			visi_ind_3D.at<int>(i+n) = 0;
			visi_ind_3D.at<int>(i+2*n) = 0;

			visi_count--;
		}
	}

	// As not all landmarks might be visible, subsample the Mean and principal component matrices
	cv::Mat_<float> M(visi_count * 3, mean_shape.cols, 0.0);
	cv::Mat_<float> V(visi_count * 3, princ_comp.cols, 0.0);
	visi_count = 0;
	for (int i = 0; i < n * 3; ++i)
	{
		if (visi_ind_3D.at<int>(i) == 1)
		{
			this->mean_shape.row(i).copyTo(M.row(visi_count));
			this->princ_comp.row(i).copyTo(V.row(visi_count));
			visi_count++;
		}
	}

	cv::Mat_<float> m_old = this->mean_shape.clone();
	cv::Mat_<float> v_old = this->princ_comp.clone();

	this->mean_shape = M;
	this->princ_comp = V;

	// The new number of points
	n  = M.rows / 3;

	// Extract the relevant landmark locations
	cv::Mat_<float> landmark_locs_vis(n*2, 1, 0.0f);
	int k = 0;
	for(int i = 0; i < visi_ind_2D.rows; ++i)
	{
		if(visi_ind_2D.at<int>(i) == 1)
		{
			landmark_locs_vis.at<float>(k) = landmark_locations.at<float>(i);
			k++;
		}		
	}

	// Compute the initial global parameters
	float min_x, max_x, min_y, max_y;
	ExtractBoundingBox(landmark_locs_vis, min_x, max_x, min_y, max_y);

	float width = abs(min_x - max_x);
	float height = abs(min_y - max_y);

	cv::Rect_<float> model_bbox;
	CalcBoundingBox(model_bbox, cv::Vec6f(1.0, 0.0, 0.0, 0.0, 0.0, 0.0), cv::Mat_<float>(this->NumberOfModes(), 1, 0.0));

	cv::Rect_<float> bbox(min_x, min_y, width, height);

	float scaling = ((width / model_bbox.width) + (height / model_bbox.height)) / 2.0f;
        
	cv::Vec3f rotation_init = rotation;
	cv::Matx33f R = Utilities::Euler2RotationMatrix(rotation_init);
	cv::Vec2f translation((min_x + max_x) / 2.0f, (min_y + max_y) / 2.0f);
    
	cv::Mat_<float> loc_params(this->NumberOfModes(),1, 0.0);
	cv::Vec6f glob_params(scaling, rotation_init[0], rotation_init[1], rotation_init[2], translation[0], translation[1]);

	// get the 3D shape of the object	
	cv::Mat_<float> shape_3D = M + V * loc_params;

	cv::Mat_<float> curr_shape(2*n, 1);
	
	// for every vertex
	for(int i = 0; i < n; i++)
	{
		// Transform this using the weak-perspective mapping to 2D from 3D
		curr_shape.at<float>(i  ,0) = scaling * ( R(0,0) * shape_3D.at<float>(i, 0) + R(0,1) * shape_3D.at<float>(i+n  ,0) + R(0,2) * shape_3D.at<float>(i+n*2,0) ) + translation[0];
		curr_shape.at<float>(i+n,0) = scaling * ( R(1,0) * shape_3D.at<float>(i, 0) + R(1,1) * shape_3D.at<float>(i+n  ,0) + R(1,2) * shape_3D.at<float>(i+n*2,0) ) + translation[1];
	}
		    
    float currError = cv::norm(curr_shape - landmark_locs_vis);

	cv::Mat_<float> regularisations = cv::Mat_<float>::zeros(1, 6 + m);

	float reg_factor = 1;

	// Setting the regularisation to the inverse of eigenvalues
	cv::Mat(reg_factor / this->eigen_values).copyTo(regularisations(cv::Rect(6, 0, m, 1)));
	regularisations = cv::Mat::diag(regularisations.t());

	cv::Mat_<float> WeightMatrix = cv::Mat_<float>::eye(n*2, n*2);

	int not_improved_in = 0;

	for (size_t i = 0; i < 1000; ++i)
	{
		// get the 3D shape of the object
		shape_3D = M + V * loc_params;

		shape_3D = shape_3D.reshape(1, 3);

		cv::Matx23f R_2D(R(0,0), R(0,1), R(0,2), R(1,0), R(1,1), R(1,2));

		cv::Mat_<float> curr_shape_2D = scaling * shape_3D.t() * cv::Mat(R_2D).t();
        curr_shape_2D.col(0) = curr_shape_2D.col(0) + translation(0);
		curr_shape_2D.col(1) = curr_shape_2D.col(1) + translation(1);

		curr_shape_2D = cv::Mat(curr_shape_2D.t()).reshape(1, n * 2);
		
		cv::Mat_<float> error_resid;
		cv::Mat(landmark_locs_vis - curr_shape_2D).convertTo(error_resid, CV_32F);
        
		cv::Mat_<float> J, J_w_t;
		this->ComputeJacobian(loc_params, glob_params, J, WeightMatrix, J_w_t);
        
		// projection of the meanshifts onto the jacobians (using the weighted Jacobian, see Baltrusaitis 2013)
		cv::Mat_<float> J_w_t_m = J_w_t * error_resid;

		// Add the regularisation term
		J_w_t_m(cv::Rect(0,6,1, m)) = J_w_t_m(cv::Rect(0,6,1, m)) - regularisations(cv::Rect(6,6, m, m)) * loc_params;

		cv::Mat_<float> Hessian = regularisations.clone();

		// Perform matrix multiplication in OpenBLAS (fortran call)
		float alpha1 = 1.0;
		float beta1 = 1.0;
		char N[2]; N[0] = 'N';
		sgemm_(N, N, &J.cols, &J_w_t.rows, &J_w_t.cols, &alpha1, (float*)J.data, &J.cols, (float*)J_w_t.data, &J_w_t.cols, &beta1, (float*)Hessian.data, &J.cols);

		// Above is a fast (but ugly) version of 
		// cv::Mat_<float> Hessian2 = J_w_t * J + regularisations;
		
		// Solve for the parameter update (from Baltrusaitis 2013 based on eq (36) Saragih 2011)
		cv::Mat_<float> param_update;
		cv::solve(Hessian, J_w_t_m, param_update, cv::DECOMP_CHOLESKY);

		// To not overshoot, have the gradient decent rate a bit smaller
		param_update = 0.75 * param_update;

		UpdateModelParameters(param_update, loc_params, glob_params);		
        
        scaling = glob_params[0];
		rotation_init[0] = glob_params[1];
		rotation_init[1] = glob_params[2];
		rotation_init[2] = glob_params[3];

		translation[0] = glob_params[4];
		translation[1] = glob_params[5];
        
		R = Utilities::Euler2RotationMatrix(rotation_init);

		R_2D(0,0) = R(0,0);R_2D(0,1) = R(0,1); R_2D(0,2) = R(0,2);
		R_2D(1,0) = R(1,0);R_2D(1,1) = R(1,1); R_2D(1,2) = R(1,2); 

		curr_shape_2D = scaling * shape_3D.t() * cv::Mat(R_2D).t();
        curr_shape_2D.col(0) = curr_shape_2D.col(0) + translation(0);
		curr_shape_2D.col(1) = curr_shape_2D.col(1) + translation(1);

		curr_shape_2D = cv::Mat(curr_shape_2D.t()).reshape(1, n * 2);
        
        float error = cv::norm(curr_shape_2D - landmark_locs_vis);
        
        if(0.999 * currError < error)
		{
			not_improved_in++;
			if (not_improved_in == 3)
			{
				break;
			}
		}

		currError = error;
        
	}

	out_params_global = glob_params;
	out_params_local = loc_params;
    	
	this->mean_shape = m_old;
	this->princ_comp = v_old;


}

bool PDM::Read(std::string location)
{

	std::ifstream pdmLoc(location, std::ios_base::in);
	if (!pdmLoc.is_open())
	{
		return false;
	}

	LandmarkDetector::SkipComments(pdmLoc);

	// Reading mean values
	cv::Mat_<double> mean_shape_d;
	LandmarkDetector::ReadMat(pdmLoc, mean_shape_d);
	mean_shape_d.convertTo(mean_shape, CV_32F); // Moving things to floats for speed

	LandmarkDetector::SkipComments(pdmLoc);

	// Reading principal components
	cv::Mat_<double> princ_comp_d;
	LandmarkDetector::ReadMat(pdmLoc, princ_comp_d);
	princ_comp_d.convertTo(princ_comp, CV_32F);

	LandmarkDetector::SkipComments(pdmLoc);
	
	// Reading eigenvalues	
	cv::Mat_<double> eigen_values_d;
	LandmarkDetector::ReadMat(pdmLoc, eigen_values_d);
	eigen_values_d.convertTo(eigen_values, CV_32F);

	return true;
}
