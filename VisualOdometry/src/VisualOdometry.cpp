//============================================================================
// Name        : VisualOdometry.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "VisualOdometry.h"

vector<string> get_image_path(string dir_path) {
	vector<string> filenames;
	for (directory_iterator itr(dir_path); itr != directory_iterator(); ++itr) {
		string this_path = dir_path + itr->path().filename().generic_string();
		filenames.push_back(this_path);

		//cout << itr->path().filename() << ' '; // display filename only
		//if (is_regular_file(itr->status())) {
		//    cout << " [" << file_size(itr->path()) << ']';
		//}
		//cout << '\n';
	}
	return filenames;
}

void printArray(double* arr, uint32_t count) {
	cout << "-----------------------------------" << endl;
	for (int i = 0; i < count; i++) {
		cout << "    value = " << arr[i] << endl;
	}
	cout << "-----------------------------------" << endl;
}

void constructBundleAdjustment(BundleAdjust& badj, vector<Frame*> frame_history) {
    // Get the number of cameras
    uint32_t num_cameras = frame_history.size();

    // Get the number of 3D points
    KeyFrame* kf = frame_history[0]->getKeyFrame();
    uint32_t num_3d_points = kf->get3DPoints().rows;

    // Setup the camera projection matrices
    Mat M1(kf->getProjectionMat());

    // We just duplicate the camera projection matrix (hopefully it won't change much)
    const uint32_t camera_mat_size = 12;
    double *cameras = new double[num_cameras*camera_mat_size];
    cameras[ 0] = M1.at<double>(0, 0);
    cameras[ 1] = M1.at<double>(0, 1);
    cameras[ 2] = M1.at<double>(0, 2);
    cameras[ 3] = M1.at<double>(0, 3);
    cameras[ 4] = M1.at<double>(1, 0);
    cameras[ 5] = M1.at<double>(1, 1);
    cameras[ 6] = M1.at<double>(1, 2);
    cameras[ 7] = M1.at<double>(1, 3);
    cameras[ 8] = M1.at<double>(2, 0);
    cameras[ 9] = M1.at<double>(2, 1);
    cameras[10] = M1.at<double>(2, 2);
    cameras[11] = M1.at<double>(2, 3);

    cv::Mat K = M1(Range(0, 3), Range(0, 3));

    for(int i=1;i<num_cameras;i++) {
        cv::Mat Rt = frame_history[i]->getPose();
        //cout << "K = " << K << endl;
        //cout << "Rt = " << Rt << endl;

        cv::Mat projMtx;
        projMtx = K*Rt(Range(0, 3), Range(0, 4));

        //cout << "projMtx = " << projMtx << endl;

        //memcpy(&cameras[camera_mat_size*i], M1.data, sizeof(double)*camera_mat_size);
        cameras[12*i+0] = projMtx.at<double>(0, 0);
        cameras[12*i+1] = projMtx.at<double>(0, 1);
        cameras[12*i+2] = projMtx.at<double>(0, 2);
        cameras[12*i+3] = projMtx.at<double>(0, 3);
        cameras[12*i+4] = projMtx.at<double>(1, 0);
        cameras[12*i+5] = projMtx.at<double>(1, 1);
        cameras[12*i+6] = projMtx.at<double>(1, 2);
        cameras[12*i+7] = projMtx.at<double>(1, 3);
        cameras[12*i+8] = projMtx.at<double>(2, 0);
        cameras[12*i+9] = projMtx.at<double>(2, 1);
        cameras[12*i+10] = projMtx.at<double>(2, 2);
        cameras[12*i+11] = projMtx.at<double>(2, 3);
    }

    //printArray(cameras, camera_mat_size*num_cameras);

    // Keep a copy to check later what changed
    double *initial_cameras = new double[num_cameras*camera_mat_size];
    memcpy(initial_cameras, cameras, sizeof(double)*camera_mat_size*num_cameras);

    // Setup the 3D points
    Mat pts3d_mat = kf->get3DPoints();
    double* pts3d = new double[pts3d_mat.rows*3];
    frame_history[0]->getCorrect3DPointOrdering(pts3d);

    double* initial_pts3d = new double[pts3d_mat.rows*3];
    //memcpy(pts3d, pts3d_mat.data, sizeof(double)*pts3d_mat.rows*3);
    memcpy(initial_pts3d, pts3d_mat.data, sizeof(double)*pts3d_mat.rows*3);

    // Setup the observed keypoints in the images corresponding to the above 3D points
    uint32_t stride = num_3d_points*2;
    uint32_t total_size = num_cameras*stride;
    double* pts2d = new double[total_size];

    for(int i=0;i<num_cameras;i++) {
        Frame* f = frame_history[i];
        double* ptr = &(pts2d[i*stride]);
        f->getObservedCorrespondingTo3DPoints(ptr);
    }

    // Now that we've gathered all data, pass it onto bundle adjustment
    badj.setCameraCount(num_cameras);
    badj.setPointCount(num_3d_points);

    for(int i=0;i<num_cameras;i++) {
        badj.setInitialCameraEstimate(i, &cameras[camera_mat_size*i]);
    }

    for(int i=0;i<num_3d_points;i++) {
        badj.setInitialPoint3d(i, &pts3d[i*3]);
    }

    for(int i=0;i<num_cameras;i++) {
        for(int j=0;j<num_3d_points;j++) {
            badj.setInitialPoint2d(i, j, pts2d[i*stride+2*j+0], pts2d[i*stride+2*j+1]);
        }
    }

    return;
}

int main(int argc, char* argv[]) {
	google::InitGoogleLogging(argv[0]);
#if defined(IS_MAC)
	string path = "/Users/jaiprakashgogi/workspace/visualodometry/dataset/dataset/sequences/00/image_0/";
#else
	string path = "/media/usinha/Utkarsh's HDD/datasets/kitti/00/image_0/";
#endif
	vector<string> filenames = get_image_path(path);

	// Initialize Viz
	viz::Viz3d myWindow("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

    Mat K

	//Initialize Map
	Map* GlobalMap = new Map();

	// For every frame, check if its a keyframe
	// Insert to Map if keyframe
	KeyFrame* curr_kf = NULL;
	KeyFrame* prev_kf = NULL;

	vector<Frame*> prev_frame_history;

	for (int i = 0; i < filenames.size(); i++) {
		cout << "Working on frame #" << i << endl;
		string f = filenames[i];
		Frame* frame = new Frame(i, f);
		frame->setKeyFrame(curr_kf);
		frame->extractFeatures();
		if (i % 5 == 0) {	//every 5th frame is a keyframe
			prev_kf = curr_kf;
			curr_kf = new KeyFrame(i, frame);
			curr_kf->setPrevKeyFrame(prev_kf);
			cout << curr_kf->getProjectionMat() << endl;

			//curr_kf->reconstructFromPrevKF(prev_kf);
			Mat points3D = curr_kf->stereoReconstruct();
			viz::WCloud cloud_widget(points3D, viz::Color::green());
			myWindow.showWidget("3D view", cloud_widget);
			myWindow.spinOnce(1, true);
			/*
			 Mat M1 = curr_kf->getProjectionMat();
			 Mat T;
			 hconcat(Mat::eye(3,3, CV_64F), Mat::zeros(3, 1, CV_64F), T);
			 Mat K = M1(Rect(0, 0, 3, 3));
			 Affine3d cam_pose = Affine3d(T);
			 viz::WCameraPosition camPos((Matx33d) K, frame->getFrame(), 5.0, viz::Color::red());
			 myWindow.showWidget("CPW1", camPos, cam_pose);
			 myWindow.spinOnce(1, true);
			 */
			// We made a new keyframe da
			frame->setKeyFrame(curr_kf);
			GlobalMap->insertKeyFrame(curr_kf);
			GlobalMap->registerCurrentKeyFrame();
			// We start with a clean slate now
			prev_frame_history.clear();
			prev_frame_history.push_back(frame);
		} else {
			prev_frame_history.push_back(frame);

			//Frame should have keyFrame before matching
			vector<vector<Point2f>> matches = frame->matchFeatures();
			cout << "Matching features da" << endl;
			Mat T = frame->getPose();
			//Mat T = frame->getCameraPose(matches);
			Mat M1 = frame->getKeyFrame()->getProjectionMat();
#if !defined(IS_MAC)
            BundleAdjust badj;
            constructBundleAdjustment(badj, prev_frame_history);
            badj.execute();

            // Copy over the poses we calculated
            for(int yo=0;yo<prev_frame_history.size();yo++) {
                Mat T;
                badj.getAdjustedCameraMatrix(yo, T);

                prev_frame_history[yo]->setPose(T);
            }
#endif

			Mat K = M1(Rect(0, 0, 3, 3));
			Affine3d cam_pose = Affine3d(T);
			viz::WCameraPosition camPos((Matx33d) K, frame->getFrame(), 5.0, viz::Color::red());
			//myWindow.showWidget("CPW1", camPos, cam_pose);

            Mat viewer_tform(4, 4, CV_64FC1);
            viewer_tform(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_64FC1);
            viewer_tform.at<double>(0, 3) = 1;
            viewer_tform.at<double>(1, 3) = 1;
            viewer_tform.at<double>(2, 3) = 1;
            viewer_tform.at<double>(3, 3) = 1;
            Affine3d viewer_pose = Affine3d(viewer_tform) * cam_pose;
            myWindow.setViewerPose(viewer_pose);
			myWindow.spinOnce(1, true);
		}
#if defined(IS_MAC)
		if (waitKey(0) == int('q')) {
			break;
		}
#endif
	}

	return 0;
}
