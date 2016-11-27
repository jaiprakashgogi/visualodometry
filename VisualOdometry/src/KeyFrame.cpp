/*
 * KeyFrame.cpp
 *
 *  Created on: Nov 21, 2016
 *      Author: jaiprakashgogi
 */

#include "KeyFrame.h"

KeyFrame::KeyFrame(Frame* frame) {
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
					3.861448000000e+02, 0.000000000000e+00, 7.188560000000e+02,
					1.852157000000e+02, 0.000000000000e+00, 0.000000000000e+00,
					0.000000000000e+00, 1.000000000000e+00,
					0.000000000000e+00));
	cout << __func__ << endl;
}

KeyFrame::~KeyFrame() {
// TODO Auto-generated destructor stub
}

Frame* KeyFrame::getFrame() {
	return frame;
}

Mat KeyFrame::stereoReconstruct() {
	cout << __func__ << endl;
	Frame* left_frame = frame;

	string left_file = frame->getFileName();
	string key = "image_0";
	string right_file = left_file.replace(left_file.find(key), key.size(),
			"image_1");

	Frame* right_frame = new Frame(right_file);
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
