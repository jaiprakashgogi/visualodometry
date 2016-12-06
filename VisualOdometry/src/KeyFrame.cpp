/*
 * KeyFrame.cpp
 *
 *  Created on: Nov 21, 2016
 *      Author: jaiprakashgogi, Utkarsh Sinha
 */

#include "KeyFrame.h"

KeyFrame::KeyFrame(int timestamp, Frame* frame) :
		timestamp(timestamp), prev_kf(NULL) {
	// TODO Auto-generated constructor stub
	this->frame = frame;
	M1 = Mat(
			Matx34d(7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
					0.000000000000e+00, 0.000000000000e+00, 7.188560000000e+02,
					1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00,
					0.000000000000e+00, 1.000000000000e+00,
					0.000000000000e+00));
	M2 = Mat(
			Matx34d(7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
					-3.861448000000e+02, 0.000000000000e+00, 7.188560000000e+02,
					1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00,
					0.000000000000e+00, 1.000000000000e+00,
					0.000000000000e+00));
	cout << __func__ << endl;
	T = Mat::eye(4, 4, CV_32F);
}

KeyFrame::~KeyFrame() {
// TODO Auto-generated destructor stub
}

Frame* KeyFrame::getFrame() {
	return frame;
}

void KeyFrame::addFrames(Frame* _frame) {
	frameVec.push_back(_frame);
	cout << __func__ << ": " << frameVec.size() << endl;
}

Mat KeyFrame::getProjectionMat() {
	return M1;
}

Mat KeyFrame::stereoReconstruct() {
	cout << __func__ << endl;
	Frame* left_frame = frame;

	string left_file = frame->getFileName();
	string key = "image_0";
	string right_file = left_file.replace(left_file.find(key), key.size(),
			"image_1");

	Frame* right_frame = new Frame(timestamp, right_file);
	right_frame->extractFeatures();
	vector<vector<Point2f>> pts = left_frame->matchFeatures(right_frame);
// extract matches from the images
	vector<Point2f> pts1 = pts.at(0);
	vector<Point2f> pts2 = pts.at(1);

//	cout << pts1.size() << endl;
//	for(int i=0; i<pts1.size(); i++) {
//		cout << pts1.at(i).x << "x" << pts1.at(i).y << " ---" <<
//				pts2.at(i).x << "x" << pts2.at(i).y << endl;
//	}

//Triangulate points
	Mat point3DTH;
	triangulatePoints(M1, M2, Mat(pts1), Mat(pts2), point3DTH);
	point3DTH = point3DTH.t();
	cout << __LINE__ << ": " << point3DTH.size() << " " << point3DTH.channels() << endl;
	convertPointsFromHomogeneous(point3DTH, point3D);
	point3D.copyTo(point3Dglobal);
	return point3D;

}

Mat KeyFrame::get3DPoints() {
	return point3D;
}

Mat KeyFrame::get3DPointsGlobal() {
	return point3Dglobal;
}

void KeyFrame::reconstructFromPrevKF(KeyFrame *prev_kf) {
	return;
}

void KeyFrame::setPrevKeyFrame(KeyFrame* _prev_kf) {
	this->prev_kf = _prev_kf;
}

KeyFrame* KeyFrame::getPrevKeyFrame() {
	return this->prev_kf;
}

Mat KeyFrame::getNew3DPoints() {
	vector<Mat> result;
	vector<Point3f> comm_pt1;
	vector<Point3f> comm_pt2;

	Frame* curr_kf_frame = this->getFrame();
	Frame* prev_kf_frame = this->prev_kf->getFrame();

	Mat pts_curr = get3DPointsGlobal();
	Mat pts_prev = prev_kf->get3DPointsGlobal();

	vector<DMatch> curr_match = curr_kf_frame->getMatches();
	vector<DMatch> prev_match = prev_kf_frame->getMatches();

	vector<DMatch> keyframe_match;
	prev_kf_frame->matchFeatures(curr_kf_frame, &keyframe_match);

	int size_kp_curr = curr_kf_frame->getKeyPoints().size();
	int size_kp_prev = prev_kf_frame->getKeyPoints().size();

	vector<int> has3d_curr(size_kp_curr, -1);
	vector<int> has3d_prev(size_kp_prev, -1);

	int i = 0;
	for (auto it : curr_kf_frame->getMatches()) {
		has3d_curr[it.queryIdx] = i++;
	}
	i = 0;
	for (auto it : prev_kf_frame->getMatches()) {
		has3d_prev[it.queryIdx] = i++;
	}

	for (auto it : keyframe_match) {
		int id_prev = it.queryIdx;
		int id_curr = it.trainIdx;
		if (has3d_prev[id_prev] == -1 && has3d_curr[id_curr] >= 0) {
			//comm_pt1.push_back(
		//			Point3f(pts_prev.at<float>(has3d_prev[id_prev], 0),
		//					pts_prev.at<float>(has3d_prev[id_prev], 1),
		//					pts_prev.at<float>(has3d_prev[id_prev], 2)));
			comm_pt2.push_back(
					Point3f(pts_curr.at<float>(has3d_curr[id_curr], 0),
							pts_curr.at<float>(has3d_curr[id_curr], 1),
							pts_curr.at<float>(has3d_curr[id_curr], 2)));
		}
	}

	//result.push_back(Mat(comm_pt1).reshape(1, comm_pt1.size()));
	//result.push_back(Mat(comm_pt2).reshape(1, comm_pt2.size()));
	//cout << __func__ << __LINE__ << Mat(comm_pt1).rows << "x" << Mat(comm_pt1).cols << "x" << Mat(comm_pt1).dims << endl;
	return Mat(comm_pt2);

}

vector<Mat> KeyFrame::getCommon3DPoints() {

	vector<Mat> result;
	vector<Point3f> comm_pt1;
	vector<Point3f> comm_pt2;

	Frame* curr_kf_frame = this->getFrame();
	Frame* prev_kf_frame = this->prev_kf->getFrame();

	Mat pts_curr = get3DPoints();
	Mat pts_prev = prev_kf->get3DPoints();

	vector<DMatch> curr_match = curr_kf_frame->getMatches();
	vector<DMatch> prev_match = prev_kf_frame->getMatches();

	vector<DMatch> keyframe_match;
	prev_kf_frame->matchFeatures(curr_kf_frame, &keyframe_match);

	int size_kp_curr = curr_kf_frame->getKeyPoints().size();
	int size_kp_prev = prev_kf_frame->getKeyPoints().size();

	vector<int> has3d_curr(size_kp_curr, -1);
	vector<int> has3d_prev(size_kp_prev, -1);

	int i = 0;
	for (auto it : curr_kf_frame->getMatches()) {
		has3d_curr[it.queryIdx] = i++;
	}
	i = 0;
	for (auto it : prev_kf_frame->getMatches()) {
		has3d_prev[it.queryIdx] = i++;
	}

	for (auto it : keyframe_match) {
		int id_prev = it.queryIdx;
		int id_curr = it.trainIdx;
		if (has3d_prev[id_prev] >= 0 && has3d_curr[id_curr] >= 0) {
			comm_pt1.push_back(
					Point3f(pts_prev.at<float>(has3d_prev[id_prev], 0),
							pts_prev.at<float>(has3d_prev[id_prev], 1),
							pts_prev.at<float>(has3d_prev[id_prev], 2)));
			comm_pt2.push_back(
					Point3f(pts_curr.at<float>(has3d_curr[id_curr], 0),
							pts_curr.at<float>(has3d_curr[id_curr], 1),
							pts_curr.at<float>(has3d_curr[id_curr], 2)));

		}
	}

	result.push_back(Mat(comm_pt1).reshape(1, comm_pt1.size()));
	result.push_back(Mat(comm_pt2).reshape(1, comm_pt2.size()));
	//cout << __func__ << __LINE__ << Mat(comm_pt1).rows << "x" << Mat(comm_pt1).cols << "x" << Mat(comm_pt1).dims << endl;
	return result;

}

void KeyFrame::setGlobalTransformation(Mat _T) {
	this->T = _T;
}

Mat KeyFrame::getPoseKF() {
	return T;
}

void KeyFrame::updatePoseKF() {
	if (!prev_kf) {
		cout << "Prev kF no found" << endl;
		return;
	}
	Frame* prev_key_frame = prev_kf->getFrame();
	Frame* curr_frame = frame;
	vector<DMatch> curr_matches; //
	vector<vector<Point2f>> match_1 = prev_key_frame->matchFeatures(curr_frame,
			&curr_matches);

	vector<DMatch> key_matches = prev_key_frame->getMatches();
	int size_kpts = prev_key_frame->getKeyPoints().size();

	vector<int> flag_key(size_kpts, -1);
	vector<int> flag_curr(size_kpts, -1);
	int i = 0;
	for (auto it : key_matches) {
		flag_key[it.queryIdx] = i++;
	}
	//cout << __LINE__ << " " << i << endl;
	i = 0;
	for (auto it : curr_matches) {
		flag_curr[it.queryIdx] = i++;
	}
	//cout << __LINE__ << " " << i << endl;

	// check for float or double
	vector<Point2f> corresp_2d;
	vector<Point3f> corresp_3d;
	Mat points3d = prev_kf->get3DPoints();

	//cout << __func__ << " "  << points3d.rows << " " << key_matches.size() << " " << curr_matches.size() << endl;

	vector<KeyPoint> curr_kpts = curr_frame->getKeyPoints();
	uint32_t counter = 0;
	for (int i = 0; i < size_kpts; i++) {
		if (flag_key[i] >= 0 && flag_curr[i] >= 0) {
			int id_3d = flag_key[i];
			corresp_3d.push_back(
					Point3f(points3d.at<float>(id_3d, 0),
							points3d.at<float>(id_3d, 1),
							points3d.at<float>(id_3d, 2)));
			int id_2d = flag_curr[i];
			int curr_id = curr_matches.at(id_2d).trainIdx;
			corresp_2d.push_back(
					Point2f(curr_kpts[curr_id].pt.x,
							curr_kpts[curr_id].pt.y));

			counter++;
		}
	}

	// Find the camera Pose using RANSAC PnP
	int iterationsCount = 100;        // number of Ransac iterations.
	float reprojectionError = 8.0; // maximum allowed distance to consider it an inlier.
	float confidence = 0.99;
	Mat M1 = prev_kf->getProjectionMat();
	Mat K = M1(Rect(0, 0, 3, 3));
	Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1); // vector of distortion coefficients
	Mat rvec = Mat::zeros(3, 1, CV_64FC1);     // output rotation vector
	Mat tvec = Mat::zeros(3, 1, CV_64FC1);  // output translation vector
	bool useExtrinsicGuess = false;

    Mat mat_corresp_3d = Mat(corresp_3d).reshape(3);
    Mat mat_corresp_2d = Mat(corresp_2d).reshape(2);

	solvePnPRansac(mat_corresp_3d, mat_corresp_2d, K, distCoeffs, rvec, tvec,
			useExtrinsicGuess, iterationsCount, reprojectionError, confidence);
	Mat R;
	Rodrigues(rvec, R); // R is 3x3

    cout << __func__ << "R = " << R << endl;
    cout << __func__ << "tvec = " << tvec << endl;

	R = R.t();  // rotation of inverse
	tvec = -R.t() * tvec; // translation of inverse
	T = Mat::eye(4, 4, R.type()); // T is 4x4
	T(Range(0, 3), Range(0, 3)) = R * 1; // copies R into T
	T(Range(0, 3), Range(3, 4)) = tvec * 1; // copies tvec into T

	Mat parentT = prev_kf->getPoseKF();
	parentT.convertTo(parentT, CV_64F);
	T = T * parentT;
    cout << __func__ << T << endl;
	//Convert points to homogenous
	Mat point3DH;
	convertPointsToHomogeneous(point3D, point3DH);
	point3DH = point3DH.reshape(1, point3DH.rows);
	point3DH.convertTo(point3DH, T.type());
	Mat point_3d_tmp = T*point3DH.t();
	point_3d_tmp = point_3d_tmp.t();
	point_3d_tmp.convertTo(point_3d_tmp, point3D.type());
	cout << __LINE__ << point_3d_tmp.size() << " " << point_3d_tmp.channels() << endl;
	//point_3d_tmp = point_3d_tmp.reshape(4, point3DH.rows);
	convertPointsFromHomogeneous(point_3d_tmp, point3Dglobal);

	return;

}
