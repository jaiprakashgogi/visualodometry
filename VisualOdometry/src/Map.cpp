/*
 * Map.cpp
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi, Utkarsh Sinha
 */

#include "Map.h"

Map::Map() :
		kf_count(0), cam_count(0) {
	// TODO Auto-generated constructor stub
	// Initialize Viz
	myWindow = viz::Viz3d("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

}

Map::~Map() {
	// TODO Auto-generated destructor stub
}

void Map::setMode(MODE _mode) {
	mode = _mode;
}

Map::MODE Map::getMode() {
	return mode;
}

int Map::getNumKeyFrames() {
	return mapkeyFrames.size();
}

void Map::insertKeyFrame(KeyFrame* kf) {
	mapkeyFrames.push_back(kf);
}

void Map::registerCurrentKeyFrame() {
	cout << __func__ << ": " << mapkeyFrames.size() << endl;
	if (mapkeyFrames.size() < 2) {
		cout << __func__ << ": Only one keyframe" << endl;

        Mat pts = mapkeyFrames[0]->get3DPointsGlobal();
        int npoints = pts.rows;
        int total_kpts = mapkeyFrames[0]->frame->kpts.size();
        int32_t *kf_idx_to_3d = new int32_t[total_kpts];
        memset(kf_idx_to_3d, -1, sizeof(int32_t)*total_kpts);
        vector<DMatch> matches = mapkeyFrames[0]->frame->matches;
        int i=0;
        for(auto it=matches.begin();it!=matches.end();++it) {
            DMatch match = *it;
            uint32_t idx_prev_kf = match.queryIdx;
            uint32_t idx_curr_kf = match.trainIdx;

            pt3d.push_back(Point3f(pts.at<float>(i, 0),
                                   pts.at<float>(i, 1),
                                   pts.at<float>(i, 2)));
          
            kf_idx_to_3d[idx_prev_kf] = i;
            i++; 
        }

        mapkeyFrames[0]->frame->setupGlobalCorrespondences(kf_idx_to_3d);
		return;
	}

	int curr_id = mapkeyFrames.size() - 1;
	int prev_id = mapkeyFrames.size() - 2;

	KeyFrame* curr_kf = mapkeyFrames.at(curr_id);

    KeyFrame* prev_kf = nullptr;
    if(prev_id >= 0) {
        prev_kf = mapkeyFrames.at(prev_id);
    }

	//Find common points between the two keyframes
	//vector<Mat> _points3d = curr_kf->getCommon3DPoints();
	//Mat T = getTfromCommon3D(_points3d);
	//curr_kf->setGlobalTransformation(T);

    Mat new_pts = curr_kf->getNew3DPoints();
    int npoints = new_pts.rows;
    cout << "New point size = " << new_pts.size() << endl;

    int not_found_count = 0;
    int found_count = 0;
    int32_t *kf_idx_to_3d = new int32_t[npoints];
    for(int i=0;i<npoints;i++) {
        float type = new_pts.at<float>(i, 0);

        if(type == 1) {
            // Type = 1 is a new point
            pt3d.push_back(Point3f(new_pts.at<float>(i, 1),
                                   new_pts.at<float>(i, 2),
                                   new_pts.at<float>(i, 3)));
            kf_idx_to_3d[i] = pt3d.size() - 1;
        } else if(type == 2) {
            // Type = 2 is an existing point
            if(prev_kf == nullptr) {
                cout << "PANIC PANIC PANIC - we shouldn't reach here if there doesn't exist a previous keyframe!" << endl;
                continue;
            }

            int32_t prev_kpt_id = (int32_t)new_pts.at<float>(i, 1);
            kf_idx_to_3d[i] = prev_kf->frame->point_cloud_correspondence[prev_kpt_id];

            // Find the match between the previous keyframe and this point
        } else {
            kf_idx_to_3d[i] = -1;
        }
    }

    curr_kf->frame->setupGlobalCorrespondences(kf_idx_to_3d);

    cout << "Found count = " << found_count << endl;
    cout << "Not found count = " << not_found_count << endl;

	return;
}

Mat Map::getTfromCommon3D(vector<Mat> _points3d) {
	cout << __func__ << ": E" << endl;
	Mat qi = _points3d.at(0);
	Mat pi = _points3d.at(1);

	Mat qi_mean(1, qi.cols, qi.type());
	Mat pi_mean(1, pi.cols, pi.type());
	reduce(qi, qi_mean, 0, CV_REDUCE_AVG);
	reduce(pi, pi_mean, 0, CV_REDUCE_AVG);

	Mat qi_mean_repeat(qi.rows, qi.cols, qi.type());
	Mat pi_mean_repeat(pi.rows, pi.cols, pi.type());
	repeat(qi_mean, qi.rows, 1, qi_mean_repeat);
	repeat(pi_mean, pi.rows, 1, pi_mean_repeat);

	Mat yi(qi.rows, qi.cols, qi.type());
	Mat xi(pi.rows, pi.cols, pi.type());
	subtract(qi, qi_mean_repeat, yi);
	subtract(pi, pi_mean_repeat, xi);

	Mat S = xi.t() * yi;
	SVD svd(S);
	Mat R = svd.vt.t() * svd.u.t();
	if (determinant(R) == -1) {
		cout << "******* det = -1 **********" << endl;
		Mat W = Mat::eye(3, 3, R.type());
		W.at<float>(3, 3) = determinant(R);
		R = svd.vt.t() * W * svd.u.t();
	}

	Mat _t = qi_mean.t() - R * pi_mean.t();
	Mat T = Mat::eye(4, 4, CV_64F);
	T(Range(0, 3), Range(0, 3)) = R * 1.0;
	T(Range(0, 3), Range(3, 4)) = _t * 1.0;
	cout << __LINE__ << ": R=" << R << " det(R): " << determinant(R) << endl;
	cout << __LINE__ << _t << endl;
	cout << __func__ << ": X" << endl;
	//cout << __LINE__ << T << R << _t << endl;
	return T;
}

void Map::renderCurrentKF() {
	int current_id = mapkeyFrames.size() - 1;
	//Mat points3D = mapkeyFrames.at(current_id)->get3DPointsGlobal();

	if (pt3d.size() == 0) {
		return;
	}

	Mat points3D(pt3d);

	viz::WCloud cloud_widget(points3D, viz::Color::green());
	myWindow.showWidget("3D view", cloud_widget);
	myWindow.spinOnce(1, true);
}

void Map::incrementTimestamp() {
	this->frame_counter++;
}

void Map::setViewerPose(Affine3d viewer_pose) {
	myWindow.setViewerPose(viewer_pose);
}

void Map::renderCurrentCamera(viz::WCameraPosition camPos, Affine3d cam_pose) {
	string cam_name = "CP" + to_string(cam_count);
	cam_count++;
	myWindow.showWidget(cam_name, camPos, cam_pose);
	myWindow.spinOnce(1, true);
}

void Map::renderPointCloud(Mat points3D) {
	cout << __func__ << endl;
	viz::WCloud cloud_widget(points3D, viz::Color::green());
	myWindow.showWidget("3D view", cloud_widget);
	myWindow.spinOnce(1, true);
	waitKey(0);
}

void Map::renderKFCameras() {
	cout << __func__ << mapkeyFrames.size() << endl;
	int kf_id = mapkeyFrames.size() - 1;
	KeyFrame* curr_kf = mapkeyFrames.at(kf_id);
	//cout << curr_kf->getProjectionMat() << endl;
	//cout << curr_kf->getProjectionMat2() << endl;
	Mat M1 = curr_kf->getProjectionMat();
	Mat M2 = curr_kf->getProjectionMat2();
	Mat K = M1(Rect(0, 0, 3, 3));
	Mat T1 = K.inv() * M1;
	Mat T2 = K.inv() * M2;
	Mat Tl = Mat::eye(4, 4, T1.type());
	Mat Tr = Mat::eye(4, 4, T2.type());
	Tl(Range(0, 3), Range(0, 4)) = T1;
	Tr(Range(0, 3), Range(0, 4)) = T2;

	cout << T1 << T2 << endl;

	Affine3d cam_pose_l = Affine3d(Tl);
	viz::WCameraPosition camPos_l((Matx33d) K, 5.0, viz::Color::red());
	renderCurrentCamera(camPos_l, cam_pose_l);
	Affine3d cam_pose_r = Affine3d(Tr);
	viz::WCameraPosition camPos_r((Matx33d) K, 5.0, viz::Color::yellow());
	renderCurrentCamera(camPos_r, cam_pose_r);
	//setViewerPose(cam_pose_r);
	//waitKey(0);

	//myWindow.showWidget("left", camPos_l, cam_pose_l);
	//myWindow.showWidget("right", camPos_r, cam_pose_r);
	//myWindow.spinOnce(1, true);

}
