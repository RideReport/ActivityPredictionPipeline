SNAME = $(shell uname -s)

# location of the Python header files

PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)

# location of the Boost Python include files and library

LFLAGS_COMMON = -L/usr/local/lib -L/usr/local/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/lib -lboost_python -lpython$(PYTHON_VERSION) -lopencv_core -lopencv_ml -lm

ifeq ($(SNAME), Linux)
LFLAGS = $(LFLAGS_COMMON) -Wl,--export-dynamic
endif
ifeq ($(SNAME), Darwin)
LFLAGS = $(LFLAGS_COMMON) -Wl -framework Accelerate
endif

CC = g++
CFLAGS = -g -std=c++11

COMPILE = $(CC) $(CFLAGS) -I$(PYTHON_INCLUDE) -I/usr/local/include -I/usr/local/Frameworks/Python.framework/Headers -fPIC -o $@ -c $<

.PHONY: all

ifeq ($(SNAME), Linux)

all: rr_mode_classification_opencv.so opencv_fft.so utilityadapter.so

endif

ifeq ($(SNAME), Darwin)

all: rr_mode_classification_opencv.so rr_mode_classification_apple.so apple_fft.so opencv_fft.so

rr_mode_classification_apple.so: rr_mode_classification_apple.oo randomforestmanager.oo fftmanager.oo
	$(CC) $^ -shared $(LFLAGS) -o $@

apple_fft.so: apple_fft.oo fftmanager.oo
	$(CC) $^ -shared $(LFLAGS) -o $@

endif

rr_mode_classification_opencv.so: rr_mode_classification_opencv.oo randomforestmanager.oo fftmanager_opencv.oo
	$(CC) $^ -shared $(LFLAGS) -o $@

fftw_fft.so: fftw_fft.oo fftmanager_fftw.oo
	$(CC) $^ -shared $(LFLAGS) -lfftw3 -o $@

opencv_fft.so: opencv_fft.oo fftmanager_opencv.oo
	$(CC) $^ -shared $(LFLAGS) -o $@

utilityadapter.so: utilityadapter.oo
	$(CC) $^ -shared $(LFLAGS) -o $@

rr_mode_classification_apple.oo: rr_mode_classification.cpp ActivityPredictor/RandomForestManager.h
	$(COMPILE) -DPYTHON_MODULE_NAME=rr_mode_classification_apple

rr_mode_classification_opencv.oo: rr_mode_classification.cpp ActivityPredictor/RandomForestManager.h
	$(COMPILE) -DPYTHON_MODULE_NAME=rr_mode_classification_opencv

randomforestmanager.oo: ActivityPredictor/RandomForestManager.cpp ActivityPredictor/RandomForestManager.h ActivityPredictor/FFTManager.h
	$(COMPILE)

fftmanager.oo: ActivityPredictor/FFTManager.cpp
	$(COMPILE)

fftmanager_fftw.oo: ActivityPredictor/FFTManager_fftw.cpp
	$(COMPILE)

fftmanager_opencv.oo: ActivityPredictor/FFTManager_opencv.cpp
	$(COMPILE)

apple_fft.oo: AppleFFTPythonAdapter.cpp AppleFFTPythonAdapter.hpp util.hpp ActivityPredictor/FFTManager.h
	$(COMPILE)

fftw_fft.oo: FFTWPythonAdapter.cpp FFTWPythonAdapter.hpp util.hpp ActivityPredictor/FFTManager.h
	$(COMPILE)

opencv_fft.oo: OpenCVFFTPythonAdapter.cpp OpenCVFFTPythonAdapter.hpp util.hpp ActivityPredictor/FFTManager.h
	$(COMPILE)

utilityadapter.oo: UtilityAdapter.cpp UtilityAdapter.hpp util.hpp ActivityPredictor/Utility.cpp ActivityPredictor/spline.h
	$(COMPILE)

.PHONY: clean install

clean:
	rm -f *.oo *.o *.so

#install:
#	cp -v *.so ../
