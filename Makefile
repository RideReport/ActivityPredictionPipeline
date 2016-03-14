# location of the Python header files
 
PYTHON_VERSION = 2.7
PYTHON_INCLUDE = /usr/include/python$(PYTHON_VERSION)
 
# location of the Boost Python include files and library
 
BOOST_INC = /usr/include
BOOST_LIB = /usr/lib/x86_86-linux-gnu
CC = g++
CFLAGS = -g -std=c++11

COMPILE = $(CC) $(CFLAGS) -I$(PYTHON_INCLUDE) -I$(BOOST_INC) -fPIC -o $@ -c $<
 
# compile mesh classes
TARGET = rr_mode_classification
 
$(TARGET).so: $(TARGET).o randomforestmanager.oo fftmanager.oo
	g++ -shared -Wl,--export-dynamic $^ -L$(BOOST_LIB) -L/usr/lib/python$(PYTHON_VERSION)/config -lboost_python -lpython$(PYTHON_VERSION) -lopencv_core -lopencv_ml -lfftw3 -lm -o $(TARGET).so
	
$(TARGET).o: $(TARGET).cpp
	$(COMPILE)

randomforestmanager.oo: RandomForestManager.cpp
	$(COMPILE)

fftmanager.oo: FFTManager.cpp
	$(COMPILE)

.PHONY: clean install

clean:
	rm *.oo *.o *.so

install:
	cp rr_mode_classification.so ../
