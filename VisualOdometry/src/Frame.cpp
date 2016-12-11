/*
 * Frame.cpp
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi, Utkarsh Sinha
 */

#include "Frame.h"

Frame::Frame(int timestamp, string filename) :
		timestamp(timestamp) {
	// TODO Auto-generated constructor stub
	frame = imread(filename);

	if (frame.data == nullptr) {
		cout << "Unable to read image: " << filename << endl;
	}

	this->filename = filename;
	//imshow("frame", frame);
	kf = nullptr;
}

Frame::~Frame() {
	// TODO Auto-generated destructor stub
}

Mat& Frame::getFrame() {
	//cout << __func__ << ": " << getFileName() << endl;
	return frame;
}

void Frame::extractFeatures() {
	double akaze_thresh = 3e-4;
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	akaze->detectAndCompute(frame, noArray(), this->kpts, this->desc);
	return;
}

int Frame::ratioTest(std::vector<std::vector<cv::DMatch>> &matches) {
	// Reference: https://github.com/BloodAxe/OpenCV-Tutorial/blob/master/OpenCV%20Tutorial/FeatureDetectionClass.cpp

	float ratio = 0.8f;
	int removed = 0;

	// for all matches
	for (auto matchIterator = matches.begin(); matchIterator != matches.end();
			++matchIterator) {

		// if 2 NN has been identified
		if (matchIterator->size() > 1) {
			auto match = (*matchIterator);

			// TODO is this correct? This should be < instead of >
			// check distance ratio
			if (match[0].distance / match[1].distance > ratio) {
				matchIterator->clear(); // remove match
				removed++;
			}
		} else {
			// does not have 2 neighbours
			matchIterator->clear(); // remove match
			removed++;
		}
	}
	return removed;
}

// Insert symmetrical matches in symMatches vector
void Frame::symmetryTest(const vector<vector<DMatch>>& matches1,
		const vector<vector<DMatch>>& matches2, vector<DMatch>& symMatches) {
	// for all matches image 1 -> image 2
	for (std::vector<std::vector<cv::DMatch>>::const_iterator matchIterator1 =
			matches1.begin(); matchIterator1 != matches1.end();
			++matchIterator1) {
		// ignore deleted matches
		if (matchIterator1->size() < 2)
			continue;
		// for all matches image 2 -> image 1
		for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator2 =
				matches2.begin(); matchIterator2 != matches2.end();
				++matchIterator2) {
			// ignore deleted matches
			if (matchIterator2->size() < 2)
				continue;
			// Match symmetry test
			if ((*matchIterator1)[0].queryIdx == (*matchIterator2)[0].trainIdx
					&& (*matchIterator2)[0].queryIdx
							== (*matchIterator1)[0].trainIdx) {
				// add symmetrical match
				symMatches.push_back(
						cv::DMatch((*matchIterator1)[0].queryIdx,
								(*matchIterator1)[0].trainIdx,
								(*matchIterator1)[0].distance));
				break; // next match in image 1 -> image 2
			}
		}
	}
}

Mat Frame::ransacTest(const vector<DMatch>& matches,
		const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
		vector<DMatch> &outMatches, vector<Point2f>& points1,
		vector<Point2f>& points2) {
	bool refineF = true;
	double distance = 3;
	double confidence = 0.99;
	cv::Mat fundamental;
	outMatches.clear();
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it) {
		float x, y;

		x = keypoints1[it->queryIdx].pt.x;
		y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));

		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(), 0);
	if (points1.size() > 0 && points2.size() > 0) {

		// Actually compute the fundamental matrix
		fundamental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2),
				inliers, cv::FM_RANSAC, distance, confidence);

		// Iterate over the inliers and
		std::vector<uchar>::const_iterator itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM = matches.begin();
		for (; itIn != inliers.end(); ++itIn, ++itM) {
			if (*itIn) {
				outMatches.push_back(*itM);
			}
		}
		if (refineF) {
			points1.clear();
			points2.clear();
			for (std::vector<cv::DMatch>::const_iterator it =
					outMatches.begin(); it != outMatches.end(); ++it) {
				// Get the position of left keypoints
				float x = keypoints1[it->queryIdx].pt.x;
				float y = keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x, y));
				// Get the position of right keypoints
				x = keypoints2[it->trainIdx].pt.x;
				y = keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x, y));
			}
			// Compute 8-point F from all accepted matches
			if (points1.size() > 0 && points2.size() > 0) {
				fundamental = cv::findFundamentalMat(cv::Mat(points1),
						cv::Mat(points2), cv::FM_8POINT);
			}
		}
	}
	return fundamental;
}

vector<vector<Point2f>> Frame::matchFeatures(Frame* frame2,
		vector<DMatch> *match) {
	//cout << "matchFeatures on timestamp " << timestamp << endl;

	vector<vector<Point2f>> result;
	Frame* frame1;
	if (frame2 == NULL) {
		frame2 = this;
		frame1 = kf->getFrame();
	} else {
		frame1 = this;
	}
	if (frame2 == NULL) {
		cout << "Frame does not have KeyFrame" << endl;
		return result;
	}

	vector<KeyPoint> kpts1 = frame1->getKeyPoints();
	Mat desc1 = frame1->getDesc();
	vector<KeyPoint> kpts2 = frame2->getKeyPoints();
	Mat desc2 = frame2->getDesc();

	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch>> matches1;
	matcher.knnMatch(desc1, desc2, matches1, 3);

	vector<vector<DMatch>> matches2;
	matcher.knnMatch(desc2, desc1, matches2, 2);

	int removed = ratioTest(matches1);
	removed = ratioTest(matches2);

	vector<DMatch> symMatches;
	symmetryTest(matches1, matches2, symMatches);

	// 5. Validate matches using RANSAC
	vector<Point2f> points1;
	vector<Point2f> points2;
	vector<DMatch> good_matches;
	Mat res;
	if (match == NULL) {
		F = ransacTest(symMatches, kpts1, kpts2, matches, points1, points2);
		drawMatches(frame1->getFrame(), kpts1, frame2->getFrame(), kpts2,
				matches, res);
	} else {
		F = ransacTest(symMatches, kpts1, kpts2, *match, points1, points2);
		drawMatches(frame1->getFrame(), kpts1, frame2->getFrame(), kpts2,
				*match, res);
	}
	imshow("res", res);
	result.push_back(points1);
	result.push_back(points2);
	return result;
}

Mat& Frame::getCameraPose(vector<vector<Point2f>> pts) {
	vector<Point2f> curr_pts = pts.at(0);
	vector<Point2f> keyf_pts = pts.at(1);
	Mat M1 = kf->getProjectionMat();
	Mat K = M1(Rect(0, 0, 3, 3));
	Mat E = findEssentialMat(Mat(curr_pts), Mat(keyf_pts), K);
	Mat R, t;
	recoverPose(E, Mat(curr_pts), Mat(keyf_pts), K, R, t);
	T = Mat::eye(4, 4, R.type());
	T(Range(0, 3), Range(0, 3)) = R * 1;
	T(Range(0, 3), Range(3, 4)) = t * 1;

	// Recover Absolute scale
	Frame* key_frame = kf->getFrame();
	vector<DMatch> curr_matches = matches;
	vector<DMatch> key_matches = key_frame->getMatches();
	int size_kpts = key_frame->getKeyPoints().size();

	vector<int> flag_key(size_kpts, -1);
	vector<int> flag_curr(size_kpts, -1);
	int i = 0;
	for (auto it : key_matches) {
		flag_key[it.queryIdx] = i++;
	}
	i = 0;
	for (auto it : curr_matches) {
		flag_curr[it.trainIdx] = i++;
	}

	// check for float or double
	vector<Point2f> corresp_2d_l, corresp_2d_r;
	vector<Point3f> corresp_3d;
	Mat points3d = kf->get3DPoints();

	for (int i = 0; i < size_kpts; i++) {
		if (flag_key[i] >= 0 && flag_curr[i] >= 0) {
			int id_3d = flag_key[i];
			corresp_3d.push_back(
					Point3f(points3d.at<float>(id_3d, 0),
							points3d.at<float>(id_3d, 1),
							points3d.at<float>(id_3d, 2)));
			int id_2d = flag_curr[i];
			int left_id = curr_matches.at(id_2d).queryIdx;
			int right_id = curr_matches.at(id_2d).trainIdx;
			corresp_2d_l.push_back(
					Point2f(this->kpts[left_id].pt.x,
							this->kpts[left_id].pt.y));
			corresp_2d_r.push_back(
					Point2f(this->kpts[right_id].pt.x,
							this->kpts[right_id].pt.y));
		}
	}

	// triangulate the points
	Mat point3DTH, point3DT;
	Mat M2;
	hconcat(R, t, M2);
	M2 = K * M2;
	triangulatePoints(M1, M2, Mat(corresp_2d_l), Mat(corresp_2d_r), point3DTH);
	point3DTH = point3DTH.t();
	convertPointsFromHomogeneous(point3DTH, point3DT);

	i = 0;
	Point3f prev_p = corresp_3d.at(i);
	Point3f prev_q(point3DT.at<float>(i, 0), point3DT.at<float>(i, 1),
			point3DT.at<float>(i, 2));
	//cout << "scale: " << endl;
	for (int i = 1; i < corresp_3d.size(); i++) {
		auto p = corresp_3d.at(i);
		Point3f q = Point3f(point3DT.at<float>(i, 0), point3DT.at<float>(i, 1),
				point3DT.at<float>(i, 2));

		float d1 = sqrt(
				(p.x - prev_p.x) * (p.x - prev_p.x)
						+ (p.y - prev_p.y) * (p.y - prev_p.y)
						+ (p.z - prev_p.z) * (p.z - prev_p.z));
		float d2 = sqrt(
				(q.x - prev_q.x) * (q.x - prev_q.x)
						+ (q.y - prev_q.y) * (q.y - prev_q.y)
						+ (q.z - prev_q.z) * (q.z - prev_q.z));
		cout << d1 / d2 << " ";
	}
	//cout << endl;

	//cout << __func__ << point3DT.rows << " --- " << corresp_3d.size() << endl;

	return T;
}

void Frame::getCorrect3DPointOrdering(double* ret) {
	// Get the pose of the Frame using PnP
	Frame* key_frame = kf->getFrame();

	// queryIdx = keyframe, trainIdx = inside frame
	vector<DMatch> curr_matches = matches;

	// queryIdx = left, trainIdx = right
	vector<DMatch> key_matches = key_frame->getMatches();

	int size_kpts = key_frame->getKeyPoints().size();

	vector<int> flag_key(size_kpts, -1);
	vector<int> flag_curr(size_kpts, -1);
	int i = 0;
	for (auto it : key_matches) {
		flag_key[it.queryIdx] = i++;
	}
	cout << __LINE__ << " " << i << endl;
	i = 0;
	for (auto it : curr_matches) {
		flag_curr[it.queryIdx] = i++;
	}
	cout << __LINE__ << " " << i << endl;

	// check for float or double
	vector<Point2f> corresp_2d;
	vector<Point3f> corresp_3d;
	Mat points3d = kf->get3DPoints();

	uint32_t num_pts = points3d.rows;

	cout << __func__ << " " << points3d.rows << " " << key_matches.size() << " "
			<< curr_matches.size() << endl;

	cout << "size_kpts = " << size_kpts << " and num_pts = " << num_pts << endl;

	uint32_t counter = 0;
	bool printed = true;
	for (int i = 0; i < size_kpts; i++) {
		if (flag_key[i] >= 0) {
			int id_3d = flag_key[i];
			Point3f pt3d = Point3f(points3d.at<float>(id_3d, 0),
					points3d.at<float>(id_3d, 1), points3d.at<float>(id_3d, 2));

			ret[3 * counter + 0] = pt3d.x;
			ret[3 * counter + 1] = pt3d.y;
			ret[3 * counter + 2] = pt3d.z;

			/*
			 if (flag_curr[i] >= 0) {
			 int id_2d = flag_curr[i];

			 int curr_id = curr_matches.at(id_2d).trainIdx;
			 if(this->timestamp % 5 == 0) {
			 // We're at a keyframe - consider the query image instead of the train image
			 curr_id = curr_matches.at(id_2d).queryIdx;
			 }


			 Point2f pt2d = Point2f(this->kpts[curr_id].pt.x,
			 this->kpts[curr_id].pt.y);

			 }*/
			counter++;
		}
	}

	return;
}

void Frame::getObservedCorrespondingTo3DPoints(double* ret) {
	//cout << "starting here" << endl;

	// Get the pose of the Frame using PnP
	Frame* key_frame = kf->getFrame();

	// queryIdx = keyframe, trainIdx = inside frame
	vector<DMatch> curr_matches = matches;

	// queryIdx = left, trainIdx = right
	vector<DMatch> key_matches = key_frame->getMatches();

	int size_kpts = key_frame->getKeyPoints().size();

	vector<int> flag_key(size_kpts, -1);
	vector<int> flag_curr(size_kpts, -1);
	int i = 0;
	for (auto it : key_matches) {
		flag_key[it.queryIdx] = i++;
	}
	//cout << __LINE__ << " " << i << endl;
	i = 0;
	for (auto it : curr_matches) {
		flag_curr[it.queryIdx] = i++;
	}
	//cout << __LINE__ << " " << i << endl;

	// check for float or double
	vector<Point2f> corresp_2d;
	vector<Point3f> corresp_3d;
	Mat points3d = kf->get3DPoints();

	uint32_t num_pts = points3d.rows;

	//cout << __func__ << " "  << points3d.rows << " " << key_matches.size() << " " << curr_matches.size() << endl;

	//cout << "size_kpts = " << size_kpts << " and num_pts = " << num_pts << endl;

	uint32_t counter = 0;
	bool printed = true;
	for (int i = 0; i < size_kpts; i++) {
		if (flag_key[i] >= 0) {
			if (flag_curr[i] >= 0) {
				int id_3d = flag_key[i];
				int id_2d = flag_curr[i];

				int curr_id = curr_matches.at(id_2d).trainIdx;
				if (this->timestamp % 5 == 0) {
					// We're at a keyframe - consider the query image instead of the train image
					curr_id = curr_matches.at(id_2d).queryIdx;
				}

				Point3f pt3d = Point3f(points3d.at<float>(id_3d, 0),
						points3d.at<float>(id_3d, 1),
						points3d.at<float>(id_3d, 2));

				Point2f pt2d = Point2f(this->kpts[curr_id].pt.x,
						this->kpts[curr_id].pt.y);

				ret[2 * counter + 0] = pt2d.x;
				ret[2 * counter + 1] = pt2d.y;
			} else {
				ret[2 * counter + 0] = -1;
				ret[2 * counter + 1] = -1;
			}
			counter++;
		}
	}

	//cout << "counter = " << counter << endl;

	return;
}

void Frame::setPose(const Mat& pose) {
	if (pose.rows != 4 && pose.cols != 4) {
		cout << "Invalid pose dimensions" << endl;
		return;
	}

	T = Mat(pose);
	return;
}

int Frame::getTimeStamp() {
	return timestamp;
}

Mat& Frame::getPose() {
	if (T.data != nullptr) {
		cout << " returning the already existing pose" << endl;
		return T;
	}
	cout << __func__ << endl;
	// Get the pose of the Frame using PnP

	cout << __func__ << __LINE__ << ": " << this->getTimeStamp() << " --- KF: "
			<< kf->getFrame()->getTimeStamp() << endl;
	Frame* key_frame = kf->getFrame();
	vector<DMatch> curr_matches = matches;
	vector<DMatch> key_matches = key_frame->getMatches();
	int size_kpts = key_frame->getKeyPoints().size();

	cout << __LINE__ << " " << size_kpts << endl;
	vector<int> flag_key(size_kpts, -1);
	vector<int> flag_curr(size_kpts, -1);
	int i = 0;
	for (auto it : key_matches) {
		flag_key[it.queryIdx] = i++;
	}
	cout << __LINE__ << " " << i << endl;
	i = 0;
	for (auto it : curr_matches) {
		flag_curr[it.queryIdx] = i++;
	}
	cout << __LINE__ << " " << i << endl;

	/* for(int i = 0; i< size_kpts; i++) {
	 cout << i << " " << flag_key[i] << " -- " << flag_curr[i] << endl;
	 } */
	// check for float or double
	vector<Point2f> corresp_2d;
	vector<Point3f> corresp_3d;
	Mat points3d = kf->get3DPointsGlobal();

	//cout << __func__ <<  points3d << endl;
	//cout << __func__ << __LINE__ << ": size_kpts " << size_kpts << endl;
	//cout << __func__ << " "  << points3d.rows << " " << key_matches.size() << " " << curr_matches.size() << endl;

	uint32_t counter = 0;
	for (int i = 0; i < size_kpts; i++) {
		if (flag_key[i] >= 0 && flag_curr[i] >= 0) {
			int id_3d = flag_key[i];
			if (points3d.at<float>(id_3d, 2) > 0) {
				//making sure the camera points are in front of the camera
				corresp_3d.push_back(
						Point3f(points3d.at<float>(id_3d, 0),
								points3d.at<float>(id_3d, 1),
								points3d.at<float>(id_3d, 2)));
				int id_2d = flag_curr[i];
				int curr_id = curr_matches.at(id_2d).trainIdx;
				corresp_2d.push_back(
						Point2f(this->kpts[curr_id].pt.x,
								this->kpts[curr_id].pt.y));

				counter++;
			}
		}
	}

	// Find the camera Pose using RANSAC PnP
	int iterationsCount = 100;        // number of Ransac iterations.
	float reprojectionError = 8.0; // maximum allowed distance to consider it an inlier.
	float confidence = 0.99;
	Mat M1 = kf->getProjectionMat();
	Mat K = M1(Rect(0, 0, 3, 3));
	Mat distCoeffs = Mat::zeros(4, 1, CV_64FC1); // vector of distortion coefficients
	Mat rvec = Mat::zeros(3, 1, CV_64FC1);     // output rotation vector
	Mat tvec = Mat::zeros(3, 1, CV_64FC1);  // output translation vector
	bool useExtrinsicGuess = false;

	solvePnPRansac(Mat(corresp_3d), Mat(corresp_2d), K, distCoeffs, rvec, tvec,
			useExtrinsicGuess, iterationsCount, reprojectionError, confidence);
	Mat R;
	Rodrigues(rvec, R); // R is 3x3
	R = R.t();  // rotation of inverse
	tvec = -R * tvec; // translation of inverse
	T = Mat::eye(4, 4, R.type()); // T is 4x4
	T(Range(0, 3), Range(0, 3)) = R * 1; // copies R into T
	T(Range(0, 3), Range(3, 4)) = tvec * 1; // copies tvec into T

	cout << __func__ << ": local T : \n" << T << endl;
	Mat parentT = this->getKeyFrame()->getPoseKF();
	parentT.convertTo(parentT, CV_64F);

	//T = T * parentT;

	return T;
}

vector<KeyPoint> Frame::getKeyPoints() {
	return kpts;
}

Mat& Frame::getDesc() {
	return desc;
}
bool Frame::isKeyFrame() {
	// Find the error using Homography and decide if it's a keyFrame?
	uint32_t match_size = matches.size();
	uint32_t kpts_size = this->kf->getFrame()->kpts.size();
	double ratio = 0.25;

	bool answer = (match_size <= ratio * kpts_size);

	cout << "match_size = " << match_size << " === kpts_size = " << kpts_size
			<< endl;

	if (answer) {
		cout << " xxxxx creating a new keyframe" << endl;
	}

	return answer;
}

void Frame::setKeyFrame(KeyFrame * kf) {
	this->kf = kf;
}

string Frame::getFileName() {
	return filename;
}

vector<DMatch>& Frame::getMatches() {
	return matches;
}

KeyFrame* Frame::getKeyFrame() {
	return kf;
}
