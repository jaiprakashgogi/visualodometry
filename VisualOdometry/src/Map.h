/*
 * Map.h
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi
 */

#ifndef MAP_H_
#define MAP_H_

#include <iostream>
#include <vector>
#include "KeyFrame.h"

using namespace std;

class Map {
	vector<KeyFrame *> keyFrames;
public:
	Map();
	void InitializeMap();
	virtual ~Map();
};

#endif /* MAP_H_ */
