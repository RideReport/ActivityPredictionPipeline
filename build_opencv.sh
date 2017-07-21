#!/bin/bash

set -e -x

mkdir -p opencv_build
cd opencv_build

VIRTUALENV_PREFIX="`python -c "from distutils.sysconfig import get_python_lib; libdir = get_python_lib(); print libdir.replace('/lib/python2.7/site-packages', '');"`"

cmake ../opencv \
  -DCMAKE_INSTALL_PREFIX="$VIRTUALENV_PREFIX" \
  -DBUILD_CUDA_STUBS=OFF \
  -DBUILD_DOCS=OFF \
  -DBUILD_EXAMPLES=OFF \
  -DBUILD_JASPER=OFF \
  -DBUILD_JPEG=OFF \
  -DBUILD_OPENEXR=OFF \
  -DBUILD_PACKAGE=OFF \
  -DBUILD_PERF_TESTS=OFF \
  -DBUILD_PNG=OFF \
  -DBUILD_SHARED_LIBS=ON \
  -DBUILD_TBB=OFF \
  -DBUILD_TIFF=OFF \
  -DBUILD_WITH_DEBUG_INFO=OFF \
  -DBUILD_WITH_DYNAMIC_IPP=OFF \
  -DBUILD_ZLIB=OFF \
  -DBUILD_opencv_apps=OFF \
  -DBUILD_opencv_calib3d=OFF \
  -DBUILD_opencv_core=ON \
  -DBUILD_opencv_features2d=OFF \
  -DBUILD_opencv_flann=OFF \
  -DBUILD_opencv_highgui=OFF \
  -DBUILD_opencv_imgcodecs=OFF \
  -DBUILD_opencv_imgproc=OFF \
  -DBUILD_opencv_java=OFF \
  -DBUILD_opencv_ml=ON \
  -DBUILD_opencv_objdetect=OFF \
  -DBUILD_opencv_photo=OFF \
  -DBUILD_opencv_python2=ON \
  -DBUILD_opencv_shape=OFF \
  -DBUILD_opencv_stitching=OFF \
  -DBUILD_opencv_superres=OFF \
  -DBUILD_opencv_ts=OFF \
  -DBUILD_opencv_video=OFF \
  -DBUILD_opencv_videoio=OFF \
  -DBUILD_opencv_videostab=OFF \
  -DBUILD_opencv_world=OFF \
  -DENABLE_PRECOMPILED_HEADERS=OFF \
  -DWITH_CLP=OFF \
  -DWITH_CUBLAS=OFF \
  -DWITH_CUDA=OFF \
  -DWITH_CUFFT=OFF \
  -DWITH_EIGEN=OFF \
  -DWITH_GSTREAMER_0_10=OFF \
  -DWITH_JASPER=OFF \
  -DWITH_JPEG=OFF \
  -DWITH_NVCUVID=OFF \
  -DWITH_OPENCL=OFF \
  -DWITH_OPENCL_SVM=OFF \
  -DWITH_OPENEXR=OFF \
  -DWITH_OPENMP=OFF \
  -DWITH_PNG=OFF \
  -DWITH_PTHREADS_PF=OFF \
  -DWITH_TBB=OFF \
  -DWITH_TIFF=OFF \
  -DWITH_WEBP=OFF \
  -DWITH_IPP=OFF \
  -DWITH_GPHOTO2=OFF \
  -DWITH_IPP_A=OFF
make -j8
make install
