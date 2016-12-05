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
	convertPointsFromHomogeneous(point3DTH, point3D);
	return point3D;

}

Mat KeyFrame::get3DPoints() {
	return point3D;
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
