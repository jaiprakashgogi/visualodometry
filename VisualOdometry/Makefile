CERES_LIB_DIR=/work/software/ceres-solver/build/lib/
CERES_INCLUDE_DIR=-I/work/software/ceres-solver/build/config -I/work/software/ceres-solver/include/
EIGEN_INCLUDE_DIR=/usr/include/eigen3/
CERES_LIBRARY_DEPENDENCIES = -lceres -llapack -lblas

CC=g++
CCFLAGS=-O3 -std=c++0x -fopenmp
CCLIBS=`pkg-config --cflags --libs opencv` -lboost_system -lboost_filesystem -L$(CERES_LIB_DIR) $(CERES_LIBRARY_DEPENDENCIES) -lglog
CCINC=-I$(EIGEN_INCLUDE_DIR) $(CERES_INCLUDE_DIR)

all: linker

linker: bundleadjust.o cmusfm.o frame.o keyframe.o map.o main.o
	$(CC) $(CCFLAGS) $? -o VisualOdometry $(CCLIBS) $(CCINC)

bundleadjust.o: src/BundleAdjust.cpp
	$(CC) $(CCFLAGS) -c $? -o $@ $(CCLIBS) $(CCINC)
	
cmusfm.o: src/cmusfm.cpp
	$(CC) $(CCFLAGS) -c $? -o $@ $(CCLIBS) $(CCINC)

frame.o: src/Frame.cpp
	$(CC) $(CCFLAGS) -c $? -o $@ $(CCLIBS) $(CCINC)

keyframe.o: src/KeyFrame.cpp
	$(CC) $(CCFLAGS) -c $? -o $@ $(CCLIBS) $(CCINC)

map.o: src/Map.cpp
	$(CC) $(CCFLAGS) -c $? -o $@ $(CCLIBS) $(CCINC)

main.o: src/VisualOdometry.cpp
	$(CC) $(CCFLAGS) -c $? -o $@ $(CCLIBS) $(CCINC)

clean:
	rm -rf VisualOdometry
