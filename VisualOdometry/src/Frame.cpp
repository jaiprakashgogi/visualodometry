/*
 * Frame.cpp
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi
 */

#include "Frame.h"

Frame::Frame(string filename) {
	// TODO Auto-generated constructor stub
	frame = imread(filename);
	this->filename = filename;
	imshow("frame", frame);
	kf = NULL;
}

Frame::~Frame() {
	// TODO Auto-generated destructor stub
}

Mat& Frame::getFrame() {
	cout << __func__ << ": " << getFileName() << endl;
	return frame;
}

void Frame::extractFeatures() {
	double akaze_thresh = 3e-4;
	Ptr<AKAZE> akaze = AKAZE::create();
	akaze->setThreshold(akaze_thresh);
	akaze->detectAndCompute(frame, noArray(), kpts, desc);
	return;
}

int Frame::ratioTest(std::vector<std::vector<cv::DMatch> > &matches) {
	//Reference: https://github.com/BloodAxe/OpenCV-Tutorial/blob/master/OpenCV%20Tutorial/FeatureDetectionClass.cpp
	float ratio = 0.8f;
	int removed = 0;
	// for all matches
	for (std::vector<std::vector<cv::DMatch> >::iterator matchIterator =
			matches.begin(); matchIterator != matches.end(); ++matchIterator) {
		// if 2 NN has been identified
		if (matchIterator->size() > 1) {
			// check distance ratio
			if ((*matchIterator)[0].distance / (*matchIterator)[1].distance
					> ratio) {
				matchIterator->clear(); // remove match
				removed++;
			}
		} else { // does not have 2 neighbours
			matchIterator->clear(); // remove match
			removed++;
		}
	}
	return removed;
}

// Insert symmetrical matches in symMatches vector
void Frame::symmetryTest(const vector<vector<DMatch> >& matches1,
		const vector<vector<DMatch> >& matches2, vector<DMatch>& symMatches) {
	// for all matches image 1 -> image 2
	for (std::vector<std::vector<cv::DMatch> >::const_iterator matchIterator1 =
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
		vector<DMatch>& outMatches, vector<Point2f>& points1,
		vector<Point2f>& points2, vector<KeyPoint>& inlier1,
		vector<KeyPoint>& inlier2, vector<DMatch>& good_matches) {
	bool refineF = true;
	double distance = 3.0;
	double confidence = 0.99;
	// Convert keypoints into Point2f
	//std::vector<cv::Point2f> points1, points2;
	cv::Mat fundamental;
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it) {
		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}
	// Compute F matrix using RANSAC
	std::vector<uchar> inliers(points1.size(), 0);
	if (points1.size() > 0 && points2.size() > 0) {
		fundamental = cv::findFundamentalMat(cv::Mat(points1), cv::Mat(points2), // matching points
		inliers,       // match status (inlier or outlier)
				cv::FM_RANSAC, // RANSAC method
				distance,      // distance to epipolar line
				confidence); // confidence probability
		// extract the surviving (inliers) matches
		std::vector<uchar>::const_iterator itIn = inliers.begin();
		std::vector<cv::DMatch>::const_iterator itM = matches.begin();
		// for all matches
		for (; itIn != inliers.end(); ++itIn, ++itM) {
			if (*itIn) { // it is a valid match
				outMatches.push_back(*itM);
			}
		}
		if (refineF) {
			// The F matrix will be recomputed with
			// all accepted matches
			// Convert keypoints into Point2f
			// for final F computation
			points1.clear();
			points2.clear();
			for (std::vector<cv::DMatch>::const_iterator it =
					outMatches.begin(); it != outMatches.end(); ++it) {
				// Get the position of left keypoints
				int new_i = static_cast<int>(inlier1.size());
				float x = keypoints1[it->queryIdx].pt.x;
				float y = keypoints1[it->queryIdx].pt.y;
				points1.push_back(cv::Point2f(x, y));
				// Get the position of right keypoints
				x = keypoints2[it->trainIdx].pt.x;
				y = keypoints2[it->trainIdx].pt.y;
				points2.push_back(cv::Point2f(x, y));
				inlier1.push_back(keypoints1[it->queryIdx]);
				inlier2.push_back(keypoints1[it->trainIdx]);
				good_matches.push_back(DMatch(new_i, new_i, 0));
			}
			// Compute 8-point F from all accepted matches
			if (points1.size() > 0 && points2.size() > 0) {
				fundamental = cv::findFundamentalMat(cv::Mat(points1),
						cv::Mat(points2), // matches
						cv::FM_8POINT); // 8-point method
			}
		}
	}
	return fundamental;
}

vector<vector<Point2f>> Frame::matchFeatures(Frame* frame2) {
	vector<vector<Point2f>> result;
	bool isStereo = false;
	if (frame2 == NULL) {
		frame2 = kf->getFrame();
		//cout << __func__ << "frame2 Null - reading keyframe: " << frame2->getFileName() << endl;
	} else {
		isStereo = true;
	}
	if (frame2 == NULL) {
		cout << "Frame does not have KeyFrame" << endl;
		return result;
	}

	vector<KeyPoint> kpts2 = frame2->getKeyPoints();
	Mat desc2 = frame2->getDesc();
	BFMatcher matcher(NORM_HAMMING);
	vector<vector<DMatch> > matches1;
	matcher.knnMatch(desc, desc2, matches1, 2);

	vector<vector<DMatch> > matches2;
	matcher.knnMatch(desc2, desc, matches2, 2);

	int removed = ratioTest(matches1);
	// clean scene image -> object image matches
	removed = ratioTest(matches2);

	//vector<DMatch> symMatches;
	symmetryTest(matches1, matches2, matches);
	// 5. Validate matches using RANSAC
	vector<Point2f> points1; // output object keypoints (Point2f)
	vector<Point2f> points2;
#if 0
	vector<DMatch> matches; // output matches
	vector<KeyPoint> in1;
	vector<KeyPoint> in2;
	vector<DMatch> good_matches;
	F = ransacTest(symMatches, kpts, kpts2, matches, points1, points2, in1, in2, good_matches);
	Mat res;
	drawMatches(frame, kpts, frame2->getFrame(), kpts2, symMatches, res);
#else
	stackPoints(matches, kpts, kpts2, points1, points2);
	Mat res;
	drawMatches(frame, kpts, frame2->getFrame(), kpts2, matches, res);
	cout << __func__ << ": " << matches.size() << endl;
#endif
	imshow("res", res);
	result.push_back(points1);
	result.push_back(points2);
	return result;
}

void Frame::stackPoints(const vector<DMatch>& matches,
		const vector<KeyPoint>& keypoints1, const vector<KeyPoint>& keypoints2,
		vector<Point2f>& points1, vector<Point2f>& points2) {
	for (std::vector<cv::DMatch>::const_iterator it = matches.begin();
			it != matches.end(); ++it) {
		// Get the position of left keypoints
		float x = keypoints1[it->queryIdx].pt.x;
		float y = keypoints1[it->queryIdx].pt.y;
		points1.push_back(cv::Point2f(x, y));
		// Get the position of right keypoints
		x = keypoints2[it->trainIdx].pt.x;
		y = keypoints2[it->trainIdx].pt.y;
		points2.push_back(cv::Point2f(x, y));
	}
}

Mat& Frame::getPose() {
	//Get the pose of the Frame using PnP

	Frame* key_frame = kf->getFrame();
	imshow("KeyFrame", key_frame->getFrame());
	vector<DMatch> curr_matches = matches;
	vector<DMatch> key_matches = key_frame->getMatches();
	int size_kpts = key_frame->getKeyPoints().size();
	cout << __func__ << " : " << size_kpts << " xx " << key_matches.size()
			<< " xx " << curr_matches.size() << endl;

	vector<int> flag_key(size_kpts, -1);
	vector<int> flag_curr(size_kpts, -1);
	int i = 0;
	for (auto it : key_matches) {
		flag_key[it.queryIdx] = i++;
	}
	for (auto it : curr_matches) {
		flag_curr[it.trainIdx] = it.queryIdx;
	}

	// check for float or double
	vector<Point2f> corresp_2d;
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
			corresp_2d.push_back(
					Point2f(this->kpts[id_2d].pt.x, this->kpts[id_2d].pt.y));
		}
	}


	cout << corresp_3d.size() << " -- " << corresp_2d.size() << endl;

}

vector<KeyPoint> Frame::getKeyPoints() {
	return kpts;
}

Mat& Frame::getDesc() {
	return desc;
}
bool Frame::isKeyFrame() {
	//Find the error using Homography and decide if it's a keyFrame?
	return false;
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
