OS := $(shell uname)

ifeq ($(OS), Darwin)
LIBS_DIR += -L/usr/local/Cellar/ceres-solver/1.11.0_2/lib/
INCLUDE_DIR += -I/usr/local/Cellar/ceres-solver/1.11.0_2/include/ -I/usr/local/Cellar/eigen/3.2.8/include/eigen3/
CFLAGS  += -DIS_MAC
else
LIBS_DIR += -L/work/software/ceres-solver/build/lib/
INCLUDE_DIR += -I/usr/local/Cellar/ceres-solver/1.11.0_2/include/ -I/usr/local/Cellar/eigen/3.2.8/include/eigen3/
LIBS += -fopenmp
endif

INCLUDE_DIR += `pkg-config --cflags opencv`
LIBS += `pkg-config --libs opencv` -lboost_system -lboost_filesystem -lglog -lceres -llapack -lblas
CFLAGS += -O3 -std=c++0x

all:
	g++ $(CFLAGS) $(wildcard src/*.cpp) $(INCLUDE_DIR) -o VisualOdometry $(LIBS_DIR) $(LIBS)
clean:
	rm -rf VisualOdometry
