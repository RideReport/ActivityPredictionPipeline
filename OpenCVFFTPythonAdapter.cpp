#include "OpenCVFFTPythonAdapter.hpp"
#include "util.hpp"
#include "ActivityPredictor/FFTManager.h"
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
    if (inputVec.size() < _sampleSize) {
      throw std::runtime_error("Insufficient data; need at least sampleSize inputs");
    }
    auto outputVec = vector<float>(_sampleSize);
    ::fft((FFTManager*)_fft, inputVec.data(), _sampleSize, outputVec.data());

    return listFromVector<float>(outputVec);
}

BOOST_PYTHON_MODULE(opencv_fft)
{
    py::class_<OpenCVFFTPythonAdapter>("OpenCVFFTPythonAdapter", py::init<int>())
        .def("fft", &OpenCVFFTPythonAdapter::fft)
    ;
}
