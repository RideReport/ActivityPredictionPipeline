SNAME = $(shell uname -s)

# location of the Python header files
 
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
 
# location of the Boost Python include files and library
 
LFLAGS_COMMON = -L/usr/local/lib -L/usr/local/Frameworks/Python.framework/Versions/$(PYTHON_VERSION)/lib -lboost_python -lpython$(PYTHON_VERSION) -lopencv_core -lopencv_ml -lfftw3 -lm

ifeq ($(SNAME), Linux)
LFLAGS = $(LFLAGS_COMMON) -Wl,--export-dynamic
endif
ifeq ($(SNAME), Darwin)
LFLAGS = $(LFLAGS_COMMON) -Wl -framework Accelerate
endif

CC = g++
CFLAGS = -g -std=c++11

COMPILE = $(CC) $(CFLAGS) -I$(PYTHON_INCLUDE) -I/usr/local/include -I/usr/local/Frameworks/Python.framework/Headers -fPIC -o $@ -c $<
 
# compile mesh classes
TARGET = rr_mode_classification
 
ifeq ($(SNAME), Linux)
$(TARGET).so: $(TARGET).o randomforestmanager.oo fftmanager_fftw.oo
	$(CC) $^ -shared $(LFLAGS) -o $(TARGET).so
endif
ifeq ($(SNAME), Darwin)
$(TARGET).so: $(TARGET).o randomforestmanager.oo fftmanager.oo fftmanager_fftw.oo
	$(CC) $^ -shared $(LFLAGS) -o $(TARGET).so
endif
	
rr_mode_classification.o: rr_mode_classification.cpp RandomForestManager.h
	$(COMPILE)

randomforestmanager.oo: RandomForestManager.cpp RandomForestManager.h FFTManager.h
	$(COMPILE)

fftmanager.oo: FFTManager.cpp
	$(COMPILE)

fftmanager_fftw.oo: FFTManager_fftw.cpp
	$(COMPILE)

.PHONY: clean install

clean:
	rm *.oo *.o *.so

install:
	cp rr_mode_classification.so ../
