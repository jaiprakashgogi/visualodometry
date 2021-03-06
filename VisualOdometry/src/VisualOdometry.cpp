//============================================================================
// Name        : VisualOdometry.cpp
// Author      : 
// Version     :
// Copyright   : Your copyright notice
// Description : Hello World in C++, Ansi-style
//============================================================================

#include "VisualOdometry.h"

#define KEYFRAME_FREQ 5

#define FEATURE_AKAZE 0
#define FEATURE_ORB 1

#define USE_DYNAMIC_KEYFRAME_GENERATION true
#define BUNDLE_ADJUST_WINDOW_SIZE 5
#define DO_BUNDLE_ADJUST false

void print_opencv_version() {
    cout << "OpenCV version : " << CV_VERSION << endl;
    cout << "Major version : " << CV_MAJOR_VERSION << endl;
    cout << "Minor version : " << CV_MINOR_VERSION << endl;
    cout << "Subminor version : " << CV_SUBMINOR_VERSION << endl;

    cout << "Boost version" << BOOST_VERSION << endl;
}

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

    std::sort(filenames.begin(), filenames.end());
    return filenames;
}

void printArray(double* arr, uint32_t count) {
    cout << "-----------------------------------" << endl;
    for (int i = 0; i < count; i++) {
        cout << "    value = " << arr[i] << endl;
    }
    cout << "-----------------------------------" << endl;
}

void constructBundleAdjustment(BundleAdjust& badj,
        vector<Frame*> frame_history,
        Map* map) {
    // Get the number of cameras
    uint32_t num_cameras = frame_history.size();

    // Get the number of 3D points
    KeyFrame* kf = frame_history[0]->getKeyFrame();
    uint32_t num_3d_points = map->pt3d.size(); //kf->get3DPoints().rows;

    // Setup the camera projection matrices
    Mat M1(kf->getProjectionMat());

    // We just duplicate the camera projection matrix (hopefully it won't change much)
    const uint32_t camera_mat_size = 12;
    double *cameras = new double[num_cameras * camera_mat_size];
    cameras[0] = M1.at<double>(0, 0);
    cameras[1] = M1.at<double>(0, 1);
    cameras[2] = M1.at<double>(0, 2);
    cameras[3] = M1.at<double>(0, 3);
    cameras[4] = M1.at<double>(1, 0);
    cameras[5] = M1.at<double>(1, 1);
    cameras[6] = M1.at<double>(1, 2);
    cameras[7] = M1.at<double>(1, 3);
    cameras[8] = M1.at<double>(2, 0);
    cameras[9] = M1.at<double>(2, 1);
    cameras[10] = M1.at<double>(2, 2);
    cameras[11] = M1.at<double>(2, 3);

    cv::Mat K = M1(Range(0, 3), Range(0, 3));

    for (int i = 1; i < num_cameras; i++) {
        cv::Mat Rt_inv;
        if(frame_history[i]->is_keyframe()) {
            Rt_inv = frame_history[i]->kf->getPoseKF().inv();
        } else {
            Rt_inv = frame_history[i]->getPose().inv();
        }
        //cout << "K = " << K << endl;
        //cout << "Rt = " << Rt << endl;

        cv::Mat projMtx;
        projMtx = K * Rt_inv(Range(0, 3), Range(0, 4));

        //memcpy(&cameras[camera_mat_size*i], M1.data, sizeof(double)*camera_mat_size);
        cameras[12 * i + 0] = projMtx.at<double>(0, 0);
        cameras[12 * i + 1] = projMtx.at<double>(0, 1);
        cameras[12 * i + 2] = projMtx.at<double>(0, 2);
        cameras[12 * i + 3] = projMtx.at<double>(0, 3);
        cameras[12 * i + 4] = projMtx.at<double>(1, 0);
        cameras[12 * i + 5] = projMtx.at<double>(1, 1);
        cameras[12 * i + 6] = projMtx.at<double>(1, 2);
        cameras[12 * i + 7] = projMtx.at<double>(1, 3);
        cameras[12 * i + 8] = projMtx.at<double>(2, 0);
        cameras[12 * i + 9] = projMtx.at<double>(2, 1);
        cameras[12 * i + 10] = projMtx.at<double>(2, 2);
        cameras[12 * i + 11] = projMtx.at<double>(2, 3);
    }

    // Now that we've gathered all data, pass it onto bundle adjustment
    badj.setCameraCount(num_cameras);

    for (int i = 0; i < num_cameras; i++) {
        badj.setInitialCameraEstimate(i, &cameras[camera_mat_size * i]);
    }

    // Keep a copy to check later what changed
    double *initial_cameras = new double[num_cameras * camera_mat_size];
    memcpy(initial_cameras, cameras,
            sizeof(double) * camera_mat_size * num_cameras);

    // Setup the 3D points
    //Mat pts3d_mat = kf->get3DPoints();
    //double* pts3d = new double[pts3d_mat.rows * 3];
    //frame_history[0]->getCorrect3DPointOrdering(pts3d);

    //double* initial_pts3d = new double[pts3d_mat.rows * 3];
    //memcpy(pts3d, pts3d_mat.data, sizeof(double)*pts3d_mat.rows*3);
    //memcpy(initial_pts3d, pts3d_mat.data, sizeof(double) * pts3d_mat.rows * 3);

    double *pt_array = new double[3];
    badj.setPointCount(num_3d_points);
    cout << "point count = " << num_3d_points << endl;
    for(int i=0;i<map->pt3d.size();i++) {
        Point3f pt = map->pt3d[i];
        pt_array[0] = pt.x;
        pt_array[1] = pt.y;
        pt_array[2] = pt.z;

        badj.setInitialPoint3d(i, pt_array);
    }

    cout << "=================================" << endl;


    return;
}

int main(int argc, char* argv[]) {
    print_opencv_version();

    google::InitGoogleLogging(argv[0]);
#if defined(IS_MAC)
    string path = "/Users/jaiprakashgogi/workspace/visualodometry/dataset/dataset/sequences/00/image_0/";
#else
    string path = "/media/usinha/Utkarsh's HDD/datasets/kitti/00/image_0/";
#endif
    vector<string> filenames = get_image_path(path);

    Mat K;

    //Initialize Map
    Map* GlobalMap = new Map();
    GlobalMap->setMode(Map::MODE::MONO);

    // For every frame, check if its a keyframe
    // Insert to Map if keyframe
    KeyFrame* curr_kf = nullptr;
    KeyFrame* prev_kf = nullptr;

    vector<Frame*> prev_frame_history;

    const uint32_t max_sz = filenames.size();

    /*
     Mat viewer_tform(4, 4, CV_64FC1);
     viewer_tform(Range(0, 3), Range(0, 3)) = Mat::eye(3, 3, CV_64FC1);
     viewer_tform.at<double>(0, 3) = 1;
     viewer_tform.at<double>(1, 3) = 1;
     viewer_tform.at<double>(2, 3) = 1;
     viewer_tform.at<double>(3, 3) = 1;
     Affine3d viewer_pose = Affine3d(viewer_tform);
     GlobalMap->setViewerPose(viewer_pose); */

    bool positioned = false;
    bool first_position = false;

    int start_frame = 0;
    for (int i = start_frame; i < max_sz; i++) {
        cout << "Working on frame #" << i << endl;
        string f = filenames[i];
        Frame* frame = new Frame(i, f);

        if (i == start_frame) {
            curr_kf = new KeyFrame(i, frame);
            curr_kf->setPrevKeyFrame(nullptr);
        }

        frame->setKeyFrame(curr_kf);
        if (curr_kf) {
            curr_kf->addFrames(frame);
        }

        uint32_t start = getTickCount();
        frame->extractFeatures();
        uint32_t end = getTickCount();
        cout << "Extracting features took "
                << (end - start) / (getTickFrequency()) << "s" << endl;

        // Match features
        start = getTickCount();
        vector<vector<Point2f>> matches = frame->matchFeatures();
        end = getTickCount();
        cout << "Feature matching took " << (end - start) / (getTickFrequency())
                << "s" << endl;


        Affine3d top_down = cv::viz::makeCameraPose(Vec3d(10, 1000, 10), Vec3d(0, 0, 0), Vec3d(0, 1, 0));
        GlobalMap->setViewerPose(top_down);

        //imshow("curr_frame", frame->getFrame());
        //if(i==start_frame || frame->isKeyframeWorthy()) {
        if (GlobalMap->getMode() == Map::MODE::MONO) {

            if(i > start_frame) {
                prev_frame_history.push_back(frame);
            }

            if (i == start_frame + 1 || frame->isKeyFrame()) {
                frame->set_keyframe();
                Mat T;
                if (i != start_frame + 1)
                    T = frame->getPose();
                prev_kf = curr_kf;
                curr_kf = new KeyFrame(i, frame);
                curr_kf->setPrevKeyFrame(prev_kf);
                //Mat pts = curr_kf->reconstructFromPrevKF(matches);
                Mat pts = curr_kf->reconstructFromPrevFrame();
                GlobalMap->insertKeyFrame(curr_kf);
                if (i != start_frame + 1) {
                    curr_kf->setPoseKF(T);
                }
                cout << "registering key frame" << endl;
                GlobalMap->registerCurrentKeyFrame();
                //if(i!=start_frame) {
                //}

                // TODO uncomment to render point cloud
                //GlobalMap->renderCurrentKF();
                GlobalMap->renderKFCameras();
            } else if (frame->getKeyFrame()->has3DPoints()) {
                if (i > start_frame) {
                    prev_kf = curr_kf;
                    Mat T = frame->getPose();
                    end = getTickCount();
                    cout << "Pose capture took "
                            << (end - start) / (getTickFrequency()) << "s"
                            << endl;
                    Mat M1 = frame->getKeyFrame()->getProjectionMat();
                    Mat K = M1(Rect(0, 0, 3, 3));
                    Affine3d cam_pose = Affine3d(T);
                    viz::WCameraPosition camPos((Matx33d) K, 5.0,
                            viz::Color::red());
                    GlobalMap->renderCurrentCamera(camPos, cam_pose);
                    //GlobalMap->setViewerPose(cam_pose);

                    //if(!first_position) {
                    //    Affine3d cp = cv::viz::makeCameraPose(Vec3d(0, 0, 0), Vec3d(0, 0, 10), Vec3f(0, 1, 0));
                    //    GlobalMap->setViewerPose(cp);
                    //}
                    frame->setupGlobalCorrespondences();
                }
            }

            if(i > 0) {
                cout << "timestamp = " << i << endl;
                cout << "Before bundle adjustment" << endl;
                //frame->computeReprojectionError(GlobalMap);
            }


            while(prev_frame_history.size() > 10) {
                prev_frame_history.erase(prev_frame_history.begin());
            }

            if(prev_frame_history.size() > 5) {
                BundleAdjust badj;
                constructBundleAdjustment(badj, prev_frame_history, GlobalMap);
                badj.execute(prev_frame_history);
                badj.extractResults(prev_frame_history, GlobalMap);
            }


            while(!positioned && waitKey(1) != int(' ')) {
                GlobalMap->renderCurrentKF();
            }
            GlobalMap->saveScreenshot();
            GlobalMap->incrementTimestamp();
            positioned = true;

            /*if(i%10 == 0) {
                positioned = false;
            }*/

            if(i > 0) {
                cout << "timestamp = " << i << endl;
                cout << "After bundle adjustment" << endl;
                //frame->computeReprojectionError(GlobalMap);
            }

            /*if(i==10) {
                exit(0);
            }*/
        } else {
            if (i % KEYFRAME_FREQ == 0) {   //every 5th frame is a keyframe
                Mat T;

                if (i > start_frame) {
                    prev_kf = curr_kf;

                    curr_kf = new KeyFrame(i, frame);
                    curr_kf->setPrevKeyFrame(prev_kf);
                }
                cout << "The projection matrix is: "
                        << curr_kf->getProjectionMat() << endl;

                //curr_kf->reconstructFromPrevKF(prev_kf);
                Map::MODE mode = GlobalMap->getMode();
                cout << __func__ << ": Mode = " << mode << endl;
                Mat points3D = curr_kf->stereoReconstruct();
                frame->setKeyFrame(curr_kf);

                //update the pose of the
                curr_kf->updatePoseKF();
                GlobalMap->insertKeyFrame(curr_kf);
                GlobalMap->registerCurrentKeyFrame();
                GlobalMap->renderCurrentKF();

                // We start with a clean slate now
                prev_frame_history.clear();
                prev_frame_history.push_back(frame);

                T = curr_kf->getPoseKF();

                Mat M1 = frame->getKeyFrame()->getProjectionMat();
                Mat K = M1(Rect(0, 0, 3, 3));
                Affine3d cam_pose = Affine3d(T);
                viz::WCameraPosition camPos((Matx33d) K, 5.0,
                        viz::Color::red());
                GlobalMap->renderCurrentCamera(camPos, cam_pose);
            } else {
                prev_frame_history.push_back(frame);

                start = getTickCount();
                Mat T = frame->getPose();
                end = getTickCount();
                cout << "Pose capture took "
                        << (end - start) / (getTickFrequency()) << "s" << endl;

                Mat M1 = frame->getKeyFrame()->getProjectionMat();
#if defined(IS_MAC)
                cout << "Running bundle adjustment" << endl;
                BundleAdjust badj;
                constructBundleAdjustment(badj, prev_frame_history, GlobalMap);
                badj.execute();

                // Copy over the poses we calculated
                for (int yo = 0; yo < prev_frame_history.size(); yo++) {
                    Mat T;
                    badj.getAdjustedCameraMatrix(yo, T);

                    prev_frame_history[yo]->setPose(T);
                }
#endif
                Mat K = M1(Rect(0, 0, 3, 3));
                Affine3d cam_pose = Affine3d(T);
                viz::WCameraPosition camPos((Matx33d) K, 1.0,
                        viz::Color::yellow());
                GlobalMap->renderCurrentCamera(camPos, cam_pose);
                //GlobalMap->setViewerPose(cam_pose);

                //myWindow.setViewerPose(viewer_pose);
            }
        }

#if defined(IS_MAC)
        if (waitKey(10) == int('q')) {
            break;
        }

        imshow("oyo", frame->frame);
        while(!positioned && waitKey(1) != int(' ')) {
            GlobalMap->renderCurrentKF();
        }
        GlobalMap->incrementTimestamp();
        positioned = true;

        if(i%30 == 0) {
            positioned = false;
        }

        //if (waitKey(0) == int('q')) {
        //      break;
        //   }
#endif
    }

    return 0;
}
