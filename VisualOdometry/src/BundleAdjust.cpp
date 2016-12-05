/*
 * BundleAdjust.cpp
 *
 *  Created on: Nov 21, 2016
 *      Author: Utkarsh Sinha, jaiprakashgogi
 */

#include "BundleAdjust.h"

struct CostFunctor {
    CostFunctor(double obsx, double obsy) : observed_x(obsx), observed_y(obsy) {}

    template<typename T> bool operator()(const T* const camera, const T* const pts, T* residual) const {
        // x^{i}_j = P_i X_j
        T x0 = camera[0]*pts[0] + camera[1]*pts[1] +  camera[2]*pts[2] + camera[3];
        T x1 = camera[4]*pts[0] + camera[5]*pts[1] +  camera[6]*pts[2] + camera[7];
        T x2 = camera[8]*pts[0] + camera[9]*pts[1] + camera[10]*pts[2] + camera[11];

        if(x2 <= 1e-6) {
            residual[0] = T(0);
            return true;
        }

        //cout << "x0 = " << T(x0 / x2) << endl;
        //cout << "x1 = " << T(x1 / x2) << endl;
        //cout << "obsx = " << T(observed_x) << endl;
        //cout << "obsy = " << T(observed_y) << endl;

        // Reprojection error da - what da
        T dx = T(x0 / x2) - T(observed_x);
        T dy = T(x1 / x2) - T(observed_y);

        // Calculate the residual
        residual[0] = T(dx*dx + dy*dy);

        //cout << "dx = " << dx << endl << "dy = " << dy << endl;

        //cout << "====================================================================" << endl;

        return true;
    }

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

void BundleAdjust::execute() const {
    if(num_cameras <= 0 || num_3d_points <= 0) {
        cout << "Invalid optimization bruv" << endl;
        return;
    }

    int stride = 2*this->num_3d_points;
    uint32_t visible_count = 0;

    ceres::Problem problem;

    // Build the optimization
    for(int i=0;i<num_3d_points;i++) {
        double* point3d = &(this->point_3d[i*3]);  // the 3D point i
        for(int j=0;j<num_cameras;j++) {
            // TODO check if 3D point i is visible in camera j
            bool visible = false;

            uint32_t idx_x = j*stride+2*i+0;
            uint32_t idx_y = j*stride+2*i+1;

            if(this->point_2d[idx_x] != -1 &&
               this->point_2d[idx_y] != -1) {
                visible = true;
                visible_count++;
            }

            // No need to add this term to the optimization
            if(!visible) {
                //cout << "skipping yo -------------" << endl;
                continue;
            }

            // observation x of 3D point i in camera j
            double observed_x = this->point_2d[idx_x];

            // observation y of 3D point i in camera j
            double observed_y = this->point_2d[idx_y]; 

            double* camera = &(this->camera_matrices[j*12]);    // the camera matrix for camera j


            // Every keypoint in each image is a cost function
            ceres::CostFunction *cost_function = new ceres::AutoDiffCostFunction<CostFunctor, 1, 12, 3>(new CostFunctor(observed_x, observed_y));
            problem.AddResidualBlock(cost_function, nullptr, camera, point3d);
        }
    }


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
