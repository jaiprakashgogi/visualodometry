/*
 * KeyFrame.cpp
 *
 *  Created on: Nov 21, 2016
 *      Author: jaiprakashgogi, Utkarsh Sinha
 */

#include "KeyFrame.h"

KeyFrame::KeyFrame(int timestamp, Frame* frame) :
		timestamp(timestamp), prev_kf(nullptr), reconstructionDone(false) {
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
	cout << __func__ << __LINE__ << endl;
	T = Mat::eye(4, 4, CV_64F);
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
Mat KeyFrame::getProjectionMat2() {
	return M2;
}

vector<Frame*>& KeyFrame::getFramesList() {
	return frameVec;
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

	Mat reconstructed;
	this->frame->getFrame().copyTo(reconstructed);
	//cout << pts1.size() << endl;
	for (int i = 0; i < pts1.size(); i++) {
		circle(reconstructed, pts1.at(i), 2, Scalar(0, 255, 255));
	}
	imshow("3d points", reconstructed);

//Triangulate points
	Mat point3DTH;
	triangulatePoints(M1, M2, Mat(pts1), Mat(pts2), point3DTH);
	point3DTH = point3DTH.t();
	convertPointsFromHomogeneous(point3DTH, point3D);
	if (!prev_kf) {
		point3D.copyTo(point3Dglobal);
	}
	reconstructionDone = true;
	return point3D;

}

Mat KeyFrame::get3DPoints() {
	return point3D;
}

Mat KeyFrame::get3DPointsGlobal() {
	return point3Dglobal;
}

Mat KeyFrame::reconstructFromPrevKF(vector<vector<Point2f>> pts) {
	if (prev_kf == nullptr) {
		cout << __func__ << __LINE__ << ": No prev_kf" << endl;
		return point3D;
	}
	cout << __func__ << endl;

	//Frame* _curr_kf = getFrame();
	//Frame* _prev_kf = prev_kf->getFrame();
	//vector<vector<Point2f>> pts = frame->matchFeatures();

	vector<Point2f> kf_pts = pts.at(0);
	vector<Point2f> curr_pts = pts.at(1);

	Mat reconstructed;
	this->frame->getFrame().copyTo(reconstructed);
	//cout << pts1.size() << endl;
	for (int i = 0; i < curr_pts.size(); i++) {
		circle(reconstructed, curr_pts.at(i), 2, Scalar(0, 255, 255));
	}
	imshow("3d points", reconstructed);

	Mat K = M1(Rect(0, 0, 3, 3));
	Mat E = findEssentialMat(kf_pts, curr_pts, K);
	//Mat E = findEssentialMat(imgpts1, imgpts2, 1.0, Point2d(0,0), RANSAC, 0.999, 3, mask);
	//Mat imgpts1, imgpts2;
	//correctMatches(E, kf_pts, curr_pts, imgpts1, imgpts2);
	//recoverPose(E, imgpts1, imgpts2, R, t, 1.0, Point2d(0,0), mask);
	//cout << "E: " << E << endl;
	Mat R, t;
	recoverPose(E, kf_pts, curr_pts, K, R, t);

	T(Range(0, 3), Range(0, 3)) = R * 1.0; // copies R into T
	T(Range(0, 3), Range(3, 4)) = t * 1.0; // copies tvec into T

	Mat _M1 = Mat::eye(3, 4, M1.type());
	Mat _M2(3, 4, M2.type());
	_M2(Range(0, 3), Range(0, 3)) = R * 1;
	_M2(Range(0, 3), Range(3, 4)) = t * 1.0;

	//cout << __func__ << __LINE__ << K << _M1 << _M2 << endl;
	M1 = K * _M1;
	M2 = K * _M2;
	//cout << kf_pts.size() << "x" << imgpts1.size() << endl;

	Mat point3DTH;
	triangulatePoints(M1, M2, kf_pts, curr_pts, point3DTH);
	point3DTH = point3DTH.t();
	convertPointsFromHomogeneous(point3DTH, point3D);

	//cout << point3D << endl;
	int count = 0;
	for (int i = 0; i < point3D.rows; i++) {
		if (point3D.at<float>(i, 2) < 0.f) {
			count++;
		}
	}
	cout << __func__ << __LINE__ << ": Invalid Points : " << count << "/"
			<< point3D.rows << endl;
	//cout << point3D << endl;

	if (!prev_kf->getPrevKeyFrame()) {
		cout << "First reconstruction. Copy to Global" << endl;
		point3D.copyTo(point3Dglobal);
	}
	reconstructionDone = true;
	//return point3D;
	cout << E << R << t << endl;
	//imshow("curr_frame", getFrame()->getFrame());
	//imshow("prev_kf", prev_kf->getFrame()->getFrame());
	//waitKey(0);
	return point3D;
}

Mat KeyFrame::reconstructFromPrevFrame() {
	if (prev_kf == nullptr) {
		cout << "Prev_kf is null" << endl;
	}

	//cout << "prev frame size: " << prev_kf->getFramesList().size() << endl;
	Frame* prev_frame = prev_kf->getFramesList()[prev_kf->getFramesList().size()
			- 2];
	//cout << "prev frame id: " << prev_frame->getTimeStamp() << endl;
	//cout << "curr frame id: " << getFrame()->getTimeStamp() << endl;

	vector<vector<Point2f>> pts = getFrame()->matchFeatures(prev_frame);

	vector<Point2f> kf_pts = pts.at(0);
	vector<Point2f> curr_pts = pts.at(1);

	Mat reconstructed;
	this->frame->getFrame().copyTo(reconstructed);
	//cout << pts1.size() << endl;
	for (int i = 0; i < curr_pts.size(); i++) {
		circle(reconstructed, curr_pts.at(i), 2, Scalar(0, 255, 255));
	}
	imshow("3d points", reconstructed);

	Mat K = M1(Rect(0, 0, 3, 3));
	Mat E = findEssentialMat(kf_pts, curr_pts, K);
	//Mat E = findEssentialMat(imgpts1, imgpts2, 1.0, Point2d(0,0), RANSAC, 0.999, 3, mask);
	//Mat imgpts1, imgpts2;
	//correctMatches(E, kf_pts, curr_pts, imgpts1, imgpts2);
	//recoverPose(E, imgpts1, imgpts2, R, t, 1.0, Point2d(0,0), mask);
	//cout << "E: " << E << endl;
	Mat R, t;
	recoverPose(E, kf_pts, curr_pts, K, R, t);

	T(Range(0, 3), Range(0, 3)) = R * 1.0; // copies R into T
	T(Range(0, 3), Range(3, 4)) = t * 1.0; // copies tvec into T

	Mat _M1 = Mat::eye(3, 4, M1.type());
	Mat _M2(3, 4, M2.type());
	_M2(Range(0, 3), Range(0, 3)) = R * 1;
	_M2(Range(0, 3), Range(3, 4)) = t * 1.0;

	//cout << __func__ << __LINE__ << K << _M1 << _M2 << endl;
	M1 = K * _M1;
	M2 = K * _M2;
	//cout << kf_pts.size() << "x" << imgpts1.size() << endl;

	Mat point3DTH;
	triangulatePoints(M1, M2, kf_pts, curr_pts, point3DTH);
	point3DTH = point3DTH.t();
	convertPointsFromHomogeneous(point3DTH, point3D);

	//cout << point3D << endl;
	int count = 0;
	for (int i = 0; i < point3D.rows; i++) {
		if (point3D.at<float>(i, 2) < 0.f) {
			count++;
		}
	}
	cout << __func__ << __LINE__ << ": Invalid Points : " << count << "/"
			<< point3D.rows << endl;
	//cout << point3D << endl;

	if (!prev_kf->getPrevKeyFrame()) {
		cout << "First reconstruction. Copy to Global" << endl;
		point3D.copyTo(point3Dglobal);
	}
	reconstructionDone = true;
	//return point3D;
	//cout << E << R << t << endl;
	//imshow("curr_frame", getFrame()->getFrame());
	//imshow("prev_kf", prev_kf->getFrame()->getFrame());
	//waitKey(0);
	return point3D;
}

void KeyFrame::setPrevKeyFrame(KeyFrame* _prev_kf) {
	this->prev_kf = _prev_kf;
}

KeyFrame* KeyFrame::getPrevKeyFrame() {
	return this->prev_kf;
}

bool KeyFrame::has3DPoints() {
	return reconstructionDone;
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

void KeyFrame::setPoseKF(Mat _T) {
	cout << __func__ << __LINE__ << ": " << _T << endl;
	T = _T;
	//Convert points to homogenous
	Mat point3DH;
	convertPointsToHomogeneous(point3D, point3DH);
	point3DH = point3DH.reshape(1, point3DH.rows);
	point3DH.convertTo(point3DH, T.type());
	Mat point_3d_tmp = T * point3DH.t();
	point_3d_tmp = point_3d_tmp.t();
	point_3d_tmp.convertTo(point_3d_tmp, point3D.type());
	cout << __LINE__ << point_3d_tmp.size() << " " << point_3d_tmp.channels()
			<< endl;
	//point_3d_tmp = point_3d_tmp.reshape(4, point3DH.rows);
	convertPointsFromHomogeneous(point_3d_tmp, point3Dglobal);
}

void KeyFrame::updatePoseKF() {
	if (prev_kf == nullptr) {
		T = Mat::eye(4, 4, CV_64FC1);
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
	Mat points3d = prev_kf->get3DPointsGlobal();

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
					Point2f(curr_kpts[curr_id].pt.x, curr_kpts[curr_id].pt.y));

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

	//cout << __func__ << "R = " << R << endl;
	//cout << __func__ << "tvec = " << tvec << endl;

	R = R.t();  // rotation of inverse
	tvec = -R * tvec; // translation of inverse
	//T = Mat::eye(4, 4, R.type()); // T is 4x4
	T(Range(0, 3), Range(0, 3)) = R * 1.0; // copies R into T
	T(Range(0, 3), Range(3, 4)) = tvec * 1.0; // copies tvec into T

	cout << __func__ << ": local_kf T: " << T << endl;
	Mat parentT = prev_kf->getPoseKF();
	parentT.convertTo(parentT, CV_64F);
	//T = T * parentT;
	cout << __func__ << ": prev_kf T: " << parentT << endl;
	cout << __func__ << ": curr_kf T: " << T << endl;
	//Convert points to homogenous
	Mat point3DH;
	convertPointsToHomogeneous(point3D, point3DH);
	point3DH = point3DH.reshape(1, point3DH.rows);
	point3DH.convertTo(point3DH, T.type());
	Mat point_3d_tmp = T * point3DH.t();
	point_3d_tmp = point_3d_tmp.t();
	point_3d_tmp.convertTo(point_3d_tmp, point3D.type());
	cout << __LINE__ << point_3d_tmp.size() << " " << point_3d_tmp.channels()
			<< endl;
	//point_3d_tmp = point_3d_tmp.reshape(4, point3DH.rows);
	convertPointsFromHomogeneous(point_3d_tmp, point3Dglobal);

	return;

}
