CERES_LIB_DIR=/work/software/ceres-solver/build/lib/
CERES_INCLUDE_DIR=-I/work/software/ceres-solver/build/config -I/work/software/ceres-solver/include/
EIGEN_INCLUDE_DIR=/usr/include/eigen3/
CERES_LIBRARY_DEPENDENCIES = -lceres -llapack -lblas
all:
	g++ -O3 -std=c++0x $(wildcard src/*.cpp) -o VisualOdometry `pkg-config --cflags --libs opencv` -lboost_system -lboost_filesystem $(CERES_INCLUDE_DIR) -L$(CERES_LIB_DIR) $(CERES_LIBRARY_DEPENDENCIES) -I$(EIGEN_INCLUDE_DIR) -fopenmp -lglog

clean:
	rm -rf VisualOdometry
