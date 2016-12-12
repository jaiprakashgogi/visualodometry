/*
 * BundleAdjust.cpp
 *
 *  Created on: Nov 21, 2016
 *      Author: Utkarsh Sinha, jaiprakashgogi
 */

// TODO optimize in SE3

#include "BundleAdjust.h"

struct CostFunctor {
    CostFunctor(double obsx, double obsy) : observed_x(obsx), observed_y(obsy) {}

    template<typename T> bool operator()(const T* const camera, const T* const pts, T* residual) const {
        //cout << "pts = " << pts[0] << ", " << pts[1] << ", " << pts[2] << endl;

        // x^{i}_j = P_i X_j
        T x0 = camera[0]*pts[0] + camera[1]*pts[1] +  camera[2]*pts[2] + camera[3];
        T x1 = camera[4]*pts[0] + camera[5]*pts[1] +  camera[6]*pts[2] + camera[7];
        T x2 = camera[8]*pts[0] + camera[9]*pts[1] + camera[10]*pts[2] + camera[11];

        /*if(x2 <= 1e-6) {
            residual[0] = T(0);
            return true;
        }*/

        /*cout << "x0 = " << T(x0 / x2) << endl;
        cout << "x1 = " << T(x1 / x2) << endl;
        cout << "obsx = " << T(observed_x) << endl;
        cout << "obsy = " << T(observed_y) << endl;*/

        // Reprojection error da - what da
        T dx = T(x0 / x2) - T(observed_x);
        T dy = T(x1 / x2) - T(observed_y);

        // Calculate the residual
        residual[0] = T(dx);
        residual[1] = T(dy);

        //cout << "dx = " << dx << endl << "dy = " << dy << endl;

        //cout << "====================================================================" << endl;

        return true; }

    double observed_x;
    double observed_y;
};

BundleAdjust::BundleAdjust() {
	// TODO Auto-generated constructor stub
}

BundleAdjust::~BundleAdjust() {
	// TODO Auto-generated destructor stub
}

void BundleAdjust::allocate2dPoints() {
    if(this->num_cameras <= 0 || this->num_3d_points <= 0)
        return;

    int stride = 2*this->num_3d_points;
    int count = stride*this->num_cameras;
    this->point_2d = new double[count];

    memset(this->point_2d, (double)(-1.0), sizeof(double)*count);
}

void BundleAdjust::setCameraCount(int i) {
    this->num_cameras = i;
    this->camera_matrices = new double[12*i];

    allocate2dPoints();
}

void BundleAdjust::setPointCount(int i) {
    this->num_3d_points = i;
    this->point_3d = new double[3*this->num_3d_points];

    allocate2dPoints();
}

void BundleAdjust::setInitialCameraEstimate(int i, double* camera) {
    memcpy(&this->camera_matrices[i*12], camera, sizeof(double)*12);
}

void BundleAdjust::setInitialPoint3d(int j, double* pt) {
    if(j>this->num_3d_points) {
        return;
    }

    memcpy(&this->point_3d[j*3], pt, sizeof(double)*3);
}

void BundleAdjust::setInitialPoint2d(int cami, int ptj, double x, double y) {
    int stride = 2*this->num_3d_points;

    if(ptj>=this->num_3d_points) {
        return;
    }
    
    this->point_2d[cami*stride + ptj*2 + 0] = x;
    this->point_2d[cami*stride + ptj*2 + 1] = y;
}

void BundleAdjust::execute(vector<Frame*> frames) const {
    if(num_cameras <= 0 || num_3d_points <= 0) {
        cout << "Invalid optimization bruv" << endl;
        return;
    }

    int stride = 2*this->num_3d_points;
    uint32_t visible_count = 0;

    ceres::Problem problem;

    int counter=0;
    uint32_t num_frames = frames.size();
    //num_frames = 1;
    for(int i=0;i<num_frames;i++) {
        double* camera = &(this->camera_matrices[i*12]);    // the camera matrix for camera j

        Frame* frame = frames[i];
        uint32_t num_kpts = frame->kpts.size();
        //num_kpts = 10;
        for(int j=0;j<num_kpts;j++) {
            int idx = frame->point_cloud_correspondence[j];
            if(idx<0 || idx>this->num_3d_points) {
                continue;
            }

            double* point3d = &(this->point_3d[idx*3]);  // the 3D point i

            float observed_x = frame->kpts[j].pt.x;
            float observed_y = frame->kpts[j].pt.y;

            //cout<< "Point = " << point3d[0] << "\t" << point3d[1] << "\t" << point3d[2] << endl;
            //cout<< "observed = " << observed_x << "\t" << observed_y << endl;

            counter++;
            ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 2, 12, 3>(new CostFunctor(observed_x, observed_y));
            problem.AddResidualBlock(cost_function, nullptr, camera, point3d);
        }
    }

    cout << "number of blocks = " << counter << endl;

    // Actually solve the problem
    ceres::Solver::Options options;
    //options.minimizer_progress_to_stdout = true;
    options.num_threads = 4;
    options.max_solver_time_in_seconds = 5;
    options.use_explicit_schur_complement = true;
    options.linear_solver_type = ceres::DENSE_SCHUR;
    options.max_num_iterations = 1e9;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    cout << summary.FullReport() << endl;

    return;
}

void BundleAdjust::getAdjustedCameraMatrix(int i, cv::Mat& cam) {
    if(cam.data == nullptr) {
        cam = cv::Mat::eye(4, 4, CV_64FC1);
    }

    cam.at<double>(0, 0) = this->camera_matrices[i*12+0];
    cam.at<double>(0, 1) = this->camera_matrices[i*12+1];
    cam.at<double>(0, 2) = this->camera_matrices[i*12+2];
    cam.at<double>(0, 3) = this->camera_matrices[i*12+3];
    cam.at<double>(1, 0) = this->camera_matrices[i*12+4];
    cam.at<double>(1, 1) = this->camera_matrices[i*12+5];
    cam.at<double>(1, 2) = this->camera_matrices[i*12+6];
    cam.at<double>(1, 3) = this->camera_matrices[i*12+7];
    cam.at<double>(2, 0) = this->camera_matrices[i*12+8];
    cam.at<double>(2, 1) = this->camera_matrices[i*12+9];
    cam.at<double>(2, 2) = this->camera_matrices[i*12+10];
    cam.at<double>(2, 3) = this->camera_matrices[i*12+11];
}

void BundleAdjust::getAdjusted3DPoint(int i, cv::Point3f& pt) {
    pt.x = this->point_3d[3*i+0];
    pt.y = this->point_3d[3*i+1];
    pt.z = this->point_3d[3*i+2];
}

void BundleAdjust::extractResults(vector<Frame*> frames, Map* map) {
     Mat K(Matx33d(7.188560000000e+02, 0.000000000000e+00, 6.071928000000e+02,
                   0.000000000000e+00, 7.188560000000e+02, 1.852157000000e+02,
                   0.000000000000e+00, 0.000000000000e+00, 1.000000000000e+00));

    Mat Kinv = K.inv();
    for(int i=0;i<frames.size();i++) {
        Frame* frame = frames[i];
        Mat pose;
        getAdjustedCameraMatrix(i, pose);

        Mat new_pose(4, 4, CV_64FC1, cv::Scalar(0));
        new_pose.at<double>(3, 3) = 1;
        Mat yoyo = Kinv * pose(Range(0, 3), Range(0, 4));
        yoyo(Range(0, 3), Range(0, 4)).copyTo(new_pose(Range(0, 3), Range(0, 4)));

        frame->setPose(new_pose.inv());

        if(frame->is_keyframe() && frame->timestamp > 1) {
            frame->kf->setPoseKF(new_pose.inv());
        }
     }

     for(int i=0;i<num_3d_points;i++) {
         Point3f pt;
         getAdjusted3DPoint(i, pt);

         map->pt3d[i].x = pt.x;
         map->pt3d[i].y = pt.y;
         map->pt3d[i].z = pt.z;
     }

     cout << "done here da" << endl;
}
