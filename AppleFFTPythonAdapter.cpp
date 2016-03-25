#include "AppleFFTPythonAdapter.hpp"
#include "util.hpp"
#include "ActivityPredictor/FFTManager.h"
#include <vector>
using namespace std;

AppleFFTPythonAdapter::AppleFFTPythonAdapter(int sampleSize) {
    _fft = (void*)createFFTManager(sampleSize);
    _sampleSize = sampleSize;
}

AppleFFTPythonAdapter::~AppleFFTPythonAdapter() {
    deleteFFTManager((FFTManager*)_fft);
}

py::list AppleFFTPythonAdapter::fft(py::list input) {
    auto inputVec = vectorFromList<float>(input); 
    auto outputVec = vector<float>(_sampleSize);
    ::fft((FFTManager*)_fft, inputVec.data(), inputVec.size(), outputVec.data()) 

    list ret;
    for (float value : outputVec) {
        ret.append(value);
    }

    return ret;
}

BOOST_PYTHON_MODULE(apple_fft)
{
    py::class_<AppleFFTPythonAdapter>("AppleFFTPythonAdapter", py::init<int>())
        .def("fft", &AppleFFTPythonAdapter::fft)
    ;
}
