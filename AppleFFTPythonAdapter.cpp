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
    if (inputVec.size() < _sampleSize) {
      throw std::runtime_error("Insufficient data; need at least sampleSize inputs");
    }
    auto outputVec = vector<float>(_sampleSize);
    ::fft((FFTManager*)_fft, inputVec.data(), _sampleSize, outputVec.data());

    return listFromVector<float>(outputVec);
}

BOOST_PYTHON_MODULE(apple_fft)
{
    py::class_<AppleFFTPythonAdapter>("AppleFFTPythonAdapter", py::init<int>())
        .def("fft", &AppleFFTPythonAdapter::fft)
    ;
}
