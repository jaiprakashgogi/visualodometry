//============================================================================
// Name        : VisualOdometry.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include <iostream>

#include "cmusfm.h"
using namespace std;

int main() {
	namedWindow("LK", 1);
	string prefix = "/Users/jaiprakashgogi/workspace/mscv-nea/data/temple/";
	float intrinsic[] = { 1520.400000, 0.000000, 302.320000, 0.000000,
			1525.900000, 246.870000, 0.000000, 0.000000, 1.000000 };

	// Initialize Viz
	viz::Viz3d myWindow("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());
	viz::WLine axis(Point3f(-0.1f, -0.1f, -0.1f), Point3f(0.1f, 0.1f, 0.1f));
	axis.setRenderingProperty(viz::LINE_WIDTH, 0.1);
	myWindow.showWidget("Line Widget", axis);

	cmusfm *mSfm = new cmusfm();
	mSfm->readfiles(prefix);
	cout << __func__ << ": # of images = " << mSfm->no_images << endl;

	Mat K(3, 3, CV_32F, intrinsic);
	K.convertTo(K, CV_64F);
	mSfm->setIntrinsic(K);

	Mat img;

	for (int i = 0; i < mSfm->no_images - 1; i++) {
		img = mSfm->showKLT(i);
		imshow("LK", img);
		mSfm->point_3d = mSfm->find3D();
		//cout << point3DT << endl;

		Mat cam1_pose = mSfm->getCamerapose(0);
		Mat cam2_pose = mSfm->getCamerapose(1);
		//cout << cam1_pose << endl << cam2_pose << endl;
		Mat cam2_R = cam2_pose(Range(0, 3), Range(0, 3));
		Mat cam2_t = cam2_pose(Range(0, 3), Range(3, 4));
		Mat cam2_R_vec;
		Rodrigues(cam2_R, cam2_R_vec);
		viz::WCloud cloud_widget(mSfm->point_3d, viz::Color::green());
		myWindow.showWidget("3D view", cloud_widget);
		Vec3f cam_focal_point = Vec3f(cam2_t) + 0.1* Vec3f(cam2_R_vec);
		Affine3f cam_pose = viz::makeCameraPose(
				cam2_t, cam_focal_point ,
				cam2_R_vec);
		viz::WCameraPosition cpw(0.5); // Coordinate axes
		viz::WCameraPosition cpw_frustum(Vec2f(0.889484, 0.523599)); // Camera frustum
		myWindow.showWidget("CPW", cpw, cam_pose);
		myWindow.spinOnce(1, true);
		if ((char) waitKey(0) == 27) {
			break;
		}
	}
	return 0;
}
