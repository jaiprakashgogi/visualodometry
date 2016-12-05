/*
 * BundleAdjust.h
 *
 *  Created on: Nov 21, 2016
 *      Author: Utkarsh Sinha, jaiprakashgogi
 */

#ifndef BUNDLEADJUST_H_
#define BUNDLEADJUST_H_

#include <ceres/ceres.h>
#include <glog/logging.h>
#include <iostream>

#include <opencv2/opencv.hpp>

using namespace std;

class BundleAdjust {
public:
	BundleAdjust();
	virtual ~BundleAdjust();

    void execute() const;
    void setCameraCount(int);
    void setPointCount(int);
    void setInitialCameraEstimate(int i, double* camera);
    void setInitialPoint3d(int j, double* pt);
    void setInitialPoint2d(int cami, int ptj, double x, double y);
    void allocate2dPoints();

    void getAdjustedCameraMatrix(int i, cv::Mat& cam);

private:
    int num_cameras = 0;
    int num_3d_points = 0;
    double* camera_matrices = nullptr;
    double* point_3d = nullptr;
    double* point_2d = nullptr;
};

#endif /* BUNDLEADJUST_H_ */
