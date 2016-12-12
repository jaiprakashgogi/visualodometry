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
    uint32_t num_keyframes = mapkeyFrames.size();

	if (num_keyframes < 2) {
		cout << __func__ << ": Only two keyframes" << endl;

        KeyFrame* frame_with_recon = mapkeyFrames[0];

        Mat pts = frame_with_recon->get3DPointsGlobal();
        int npoints = pts.rows;

        int total_kpts = frame_with_recon->frame->kpts.size();
        cout << "total_kpts = " << total_kpts << endl;
        int32_t *kf_idx_to_3d = new int32_t[total_kpts];
        memset(kf_idx_to_3d, -1, sizeof(int32_t)*total_kpts);
        vector<DMatch> matches = frame_with_recon->frame->matches;
        int i=0;
        for(auto it=matches.begin();it!=matches.end();++it) {
            //cout << "inside here = " << i << endl;
            DMatch match = *it;
            uint32_t idx_prev_kf = match.queryIdx;
            uint32_t idx_curr_kf = match.trainIdx;

            pt3d.push_back(Point3f(pts.at<float>(i, 0),
                                   pts.at<float>(i, 1),
                                   pts.at<float>(i, 2)));
          
            kf_idx_to_3d[idx_prev_kf] = i;
            i++; 
        }

        cout << "Done setting up correspondences for the first frame" << endl;
        frame_with_recon->frame->setupGlobalCorrespondences(kf_idx_to_3d);
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
    cout << "===================================" << endl;

    int not_found_count = 0;
    int found_count = 0;
    cout << "sizeof kf_idx = " << npoints << endl;

    uint32_t old_ptcloud_size = pt3d.size();
    uint32_t new_ptcloud_size = pt3d.size();

    for(int i=0;i<npoints;i++) {
        float type = new_pts.at<float>(i, 0);
        if(type == 1) {
            new_ptcloud_size++;
        } else {
        }
    }

    //int32_t *kf_idx_to_3d = new int32_t[new_ptcloud_size];
    uint32_t num_kpts = curr_kf->frame->kpts.size();
    int32_t *kf_idx_to_3d = new int32_t[num_kpts];
    memset(kf_idx_to_3d, -1, sizeof(int32_t)*num_kpts);

    int new_counter = 0;
    int old_counter = 0;
    int bad_counter = 0;
    vector<DMatch> matches_curr = curr_kf->frame->matches;
    vector<DMatch> matches_prev = prev_kf->frame->matches;
    vector<DMatch> matches_keyframe = curr_kf->keyframe_match;

    cout << "matches_curr.size() = " << matches_curr.size() << endl;
    cout << "matches_prev.size() = " << matches_prev.size() << endl;
    cout << "matches_keyframe.size() = " << matches_keyframe.size() << endl;
    cout << "Size of new points = " << new_pts.rows << endl;

    for(int i=0;i<num_kpts;i++) {
        int type = (int)new_pts.at<float>(i, 0);
        Point3f given(new_pts.at<float>(i, 1),
                      new_pts.at<float>(i, 2),
                      new_pts.at<float>(i, 3));

        int id_prev;

        // New point
        switch(type) {
            case 1:
                // This is a new point - so we need to append it
                pt3d.push_back(given);

                kf_idx_to_3d[i] = pt3d.size() - 1;
                break;

            case 2:
                // This point already exists, we need to find the correspondence in the
                // previous kf_idx_to_3d
                id_prev = (int)(new_pts.at<float>(i, 4));
                kf_idx_to_3d[i] = prev_kf->frame->point_cloud_correspondence[id_prev];
                break;

            default:
                break;
        }
    }

    cout << "Done with correspondences for future frames" << endl;
    curr_kf->frame->setupGlobalCorrespondences(kf_idx_to_3d);

    /*if(matches_curr.size() != new_pts.rows) {
        cout << "Something is wrong here" << endl;
        exit(0);
    } else {
        cout << "The size was correct" << endl;
        exit(0);
    }*/

    /*for(auto it=matches_keyframe.begin();it!=matches_keyframe.end();++it) {
        DMatch match = *it;
        uint32_t idx_query = match.queryIdx;
        uint32_t idx_train = match.trainIdx;

        float type = new_pts.at<float>(idx_query, 0);
        if(type ==1) {
            cout << "There was a problem bruv = " << type << endl;
        }
    }*/

    /*for(int i=0;i<num_kpts;i++) {
        uint32_t idx_query = matches[i].queryIdx;
        uint32_t idx_train = matches[i].trainIdx;

        float type = new_pts.at<float>(idx_query, 0);
        cout << "type = " << type << endl;

        if(type == 1) {
            cout << "The first point type" << endl;

            // Type = 1 is a new point
            pt3d.push_back(Point3f(new_pts.at<float>(idx_query, 1),
                                   new_pts.at<float>(idx_query, 2),
                                   new_pts.at<float>(idx_query, 3)));

            uint32_t yoyoyo = new_pts.at<float>(idx_query, 4);
            kf_idx_to_3d[i] = pt3d.size() - 1;
            new_counter++;
        } else if(type == 2) {
            cout << "The second point type" << endl;

            // Type = 2 is an existing point
            if(prev_kf == nullptr) {
                cout << "PANIC PANIC PANIC - we shouldn't reach here if there doesn't exist a previous keyframe!" << endl;
                continue;
            }

            int32_t prev_kpt_id = (int32_t)new_pts.at<float>(i, 1);
            int32_t value = prev_kf->frame->point_cloud_correspondence[prev_kpt_id];

            cout << "value = " << value << endl;

            kf_idx_to_3d[i] = value;

            // Find the match between the previous keyframe and this point
            old_counter++;
        } else {
            cout << "The third point type" << endl;
            kf_idx_to_3d[i] = -1;
            bad_counter++;
        }
    }

    curr_kf->frame->setupGlobalCorrespondences(kf_idx_to_3d);

    cout << "new_counter = " << new_counter << endl;
    cout << "old_counter = " << old_counter << endl;
    cout << "bad_counter = " << bad_counter << endl;

    cout << "Found count = " << found_count << endl;
    cout << "Not found count = " << not_found_count << endl;*/

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

void Map::saveScreenshot() {
    char* fname = new char[256];
    sprintf(fname, "./shot-%04d.png", frame_counter);
    cout << "Saving to file = " << fname << endl;

    if(frame_counter >= 1) {
        string fn(fname);
        myWindow.saveScreenshot(fn);
    }
}

void Map::incrementTimestamp() {
	this->frame_counter++;
}

void Map::setViewerPose(Affine3d viewer_pose) {
    myWindow.setViewerPose(viewer_pose);
}

void Map::renderCurrentCamera(viz::WCameraPosition camPos, Affine3d cam_pose) {
    string cam_name = "CP" + to_string(cam_count);
    cout << "Setting the camera name = " << cam_name << endl;
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
	viz::WCameraPosition camPos_l((Matx33d) K, 5, viz::Color::red());
	renderCurrentCamera(camPos_l, cam_pose_l);
	Affine3d cam_pose_r = Affine3d(Tr);
	viz::WCameraPosition camPos_r((Matx33d) K, 5, viz::Color::yellow());
	renderCurrentCamera(camPos_r, cam_pose_r);
	//setViewerPose(cam_pose_r);
	//waitKey(0);

	//myWindow.showWidget("left", camPos_l, cam_pose_l);
	//myWindow.showWidget("right", camPos_r, cam_pose_r);
	//myWindow.spinOnce(1, true);

}
