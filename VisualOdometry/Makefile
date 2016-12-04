OS := $(shell uname)

CC=g++
CCFLAGS=-O3 -std=c++0x

INCLUDE_DIR=`pkg-config --cflags opencv`
LIBS_DIR=`pkg-config --libs opencv`
LIBS=-lceres -llapack -lblas -lboost_system -lboost_filesystem -lglog

ifeq ($(OS), Darwin)
LIBS_DIR += -L/usr/local/Cellar/ceres-solver/1.11.0_2/lib/
INCLUDE_DIR += -I/usr/local/Cellar/ceres-solver/1.11.0_2/include/ -I/usr/local/Cellar/eigen/3.2.8/include/eigen3/
CCFLAGS  += -DIS_MAC
else
LIBS_DIR += -L/work/software/ceres-solver/build/lib/
INCLUDE_DIR += -I/work/software/ceres-solver/build/config -I/work/software/ceres-solver/include/ -I/usr/include/eigen3
LIBS += -fopenmp
endif

CCINC=$(INCLUDE_DIR)
CCLIBS=$(LIBS_DIR) $(LIBS)

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
