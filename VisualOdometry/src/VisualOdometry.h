/*
 * VisualOdometry.h
 *
 *  Created on: Nov 26, 2016
 *      Author: jaiprakashgogi
 */

#ifndef VISUALODOMETRY_H_
#define VISUALODOMETRY_H_

#include <iostream>

#include "cmusfm.h"
//#include <filesystem>
#include <boost/filesystem.hpp>

using namespace std;
using namespace boost::filesystem;

vector<string> get_image_path(string dir_path) {

	vector<string> filenames;
	for (directory_iterator itr(dir_path); itr != directory_iterator(); ++itr) {

		filenames.push_back(dir_path + itr->path().filename().generic_string());
//		cout << itr->path().filename() << ' '; // display filename only
//		if (is_regular_file(itr->status()))
//			cout << " [" << file_size(itr->path()) << ']';
//		cout << '\n';
	}
	return filenames;
}

#endif /* VISUALODOMETRY_H_ */
