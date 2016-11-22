/*
 * KeyFrame.h
 *
 *  Created on: Nov 21, 2016
 *      Author: jaiprakashgogi
 */

#ifndef KEYFRAME_H_
#define KEYFRAME_H_

#include "opencv2/video/tracking.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/videoio/videoio.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/calib3d.hpp"
#include <opencv2/viz/vizcore.hpp>
#include <opencv2/calib3d/calib3d.hpp>

class KeyFrame {
public:
	KeyFrame();
	virtual ~KeyFrame();
};

#endif /* KEYFRAME_H_ */
