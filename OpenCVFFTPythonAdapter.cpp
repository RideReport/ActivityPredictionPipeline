#include "OpenCVFFTPythonAdapter.hpp"
#include "util.hpp"
#include "ActivityPredictor/FFTManager_opencv.h"
#include <vector>
using namespace std;

OpenCVFFTPythonAdapter::OpenCVFFTPythonAdapter(int sampleSize) {
    _fft = (void*) createFFTManager(sampleSize);
    _sampleSize = sampleSize;
}

OpenCVFFTPythonAdapter::~OpenCVFFTPythonAdapter() {
    deleteFFTManager((FFTManager*) _fft);
}

py::list OpenCVFFTPythonAdapter::fft(py::list input) {
    auto inputVec = vectorFromList<float>(input);
    auto outputVec = vector<float>(_sampleSize);
    ::fft((FFTManager*)_fft, inputVec.data(), inputVec.size(), outputVec.data());

    py::list ret;
    for (float value : outputVec) {
        ret.append(value);
    }

    return ret;
}

BOOST_PYTHON_MODULE(opencv_fft)
{
    py::class_<OpenCVFFTPythonAdapter>("OpenCVFFTPythonAdapter", py::init<int>())
        .def("fft", &OpenCVFFTPythonAdapter::fft)
    ;
}
