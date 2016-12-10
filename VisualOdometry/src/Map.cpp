/*
 * Map.cpp
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi, Utkarsh Sinha
 */

#include "Map.h"

Map::Map() : kf_count(0), cam_count(0) {
	// TODO Auto-generated constructor stub
	// Initialize Viz
	myWindow = viz::Viz3d("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

}

Map::~Map() {
	// TODO Auto-generated destructor stub
}

int Map::getNumKeyFrames() {
	return mapkeyFrames.size();
}

void Map::insertKeyFrame(KeyFrame* kf) {
	mapkeyFrames.push_back(kf);
}

void Map::registerCurrentKeyFrame() {
	cout << __func__ << ": " << mapkeyFrames.size() << endl;
	if(mapkeyFrames.size() < 2) {
		cout << __func__ << ": Only one keyframe" << endl;

        Mat pts = mapkeyFrames[0]->get3DPointsGlobal();
        int npoints = pts.rows;
        for(int i=0;i<npoints;i++) {
            pt3d.push_back(Point3f(pts.at<float>(i, 0),
                                   pts.at<float>(i, 1),
                                   pts.at<float>(i, 2)));
        }
		return;
	}

	int curr_id = mapkeyFrames.size() - 1;
	KeyFrame* curr_kf = mapkeyFrames.at(curr_id);

	//Find common points between the two keyframes
	//vector<Mat> _points3d = curr_kf->getCommon3DPoints();
	//Mat T = getTfromCommon3D(_points3d);
	//curr_kf->setGlobalTransformation(T);

    Mat new_pts = curr_kf->getNew3DPoints();
    cout << "New point size = " << new_pts.size() << endl;
    int npoints = new_pts.rows;
    for(int i=0;i<npoints;i++) {
        pt3d.push_back(Point3f(new_pts.at<float>(i, 0),
                               new_pts.at<float>(i, 1),
                               new_pts.at<float>(i, 2)));
    }
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
	if(determinant(R) == -1) {
		cout << "******* det = -1 **********" << endl;
		Mat W = Mat::eye(3, 3, R.type());
		W.at<float>(3,3) = determinant(R);
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

    if(pt3d.size() == 0) {
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
