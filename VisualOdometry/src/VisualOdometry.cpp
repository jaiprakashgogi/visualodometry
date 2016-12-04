//============================================================================
// Name        : VisualOdometry.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "VisualOdometry.h"
#include "Frame.h"

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



int main() {
    //BundleAdjust adj;
    //adj.execute();
    //exit(0);
    
	string path = "/media/usinha/Utkarsh's HDD/datasets/kitti/00/image_0/";
	vector<string> filenames = get_image_path(path);

	// Initialize Viz
	viz::Viz3d myWindow("Coordinate Frame");
	myWindow.showWidget("Coordinate Widget", viz::WCoordinateSystem());

	//Initialize Map

	// For every frame, check if its a keyframe
	// Insert to Map if keyframe
	KeyFrame* curr_kf = NULL;
	KeyFrame* prev_kf = NULL;

	for (int i = 0; i < filenames.size(); i++) {
        cout << "Working on frame #" << i << endl;
		string f = filenames[i];
		Frame* frame = new Frame(i, f);
		frame->setKeyFrame(curr_kf);
		frame->extractFeatures();
		if (i % 5 == 0) {
			prev_kf = curr_kf;
			curr_kf = new KeyFrame(i, frame);
			//curr_kf->reconstructFromPrevKF(prev_kf);
			Mat points3D = curr_kf->stereoReconstruct();
			viz::WCloud cloud_widget(points3D, viz::Color::green());
			myWindow.showWidget("3D view", cloud_widget);
            myWindow.spinOnce(1, true);
		} else {
			//Frame should have keyFrame before matching
			cout << i << endl;
			vector<vector<Point2f>> matches = frame->matchFeatures();
            cout << "Matching features da" << endl;
			Mat T = frame->getPose();
			//Mat T = frame->getCameraPose(matches);
			cout << T << endl;
			Mat M1 = frame->getKeyFrame()->getProjectionMat();
			Mat K = M1(Rect(0, 0, 3, 3));
			Affine3d cam_pose = Affine3d(T);
			viz::WCameraPosition camPos((Matx33d) K, frame->getFrame(), 5.0, viz::Color::red());
			myWindow.showWidget("CPW1", camPos, cam_pose);
			myWindow.spinOnce(1, true);

            Mat viewer_tform(4, 4, CV_64FC1);
            viewer_tform(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_64FC1);
            viewer_tform.at<double>(0, 3) = 1;
            viewer_tform.at<double>(1, 3) = 1;
            viewer_tform.at<double>(2, 3) = 1;
            viewer_tform.at<double>(3, 3) = 1;
            Affine3d viewer_pose = Affine3d(viewer_tform) * cam_pose;
            myWindow.setViewerPose(viewer_pose);
		}
		/*if (waitKey(0) == int('q')) {
			break;
		}*/
	}

	return 0;
}
