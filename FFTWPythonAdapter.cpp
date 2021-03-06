#include "FFTWPythonAdapter.hpp"
#include "util.hpp"
#include "ActivityPredictor/FFTManager_fftw.h"
#include <vector>
using namespace std;

FFTWPythonAdapter::FFTWPythonAdapter(int sampleSize) {
    _fft = (void*)createFFTManager(sampleSize);
    _sampleSize = sampleSize;
}

FFTWPythonAdapter::~FFTWPythonAdapter() {
    deleteFFTManager((FFTManager*)_fft);
}

py::list FFTWPythonAdapter::fft(py::list input) {
    auto inputVec = vectorFromList<float>(input); 
    auto outputVec = vector<float>(_sampleSize);
    ::fft((FFTManager*)_fft, inputVec.data(), inputVec.size(), outputVec.data());

    py::list ret;
    for (float value : outputVec) {
        ret.append(value);
    }

    return ret;
}

BOOST_PYTHON_MODULE(fftw_fft)
{
    py::class_<FFTWPythonAdapter>("FFTWPythonAdapter", py::init<int>())
        .def("fft", &FFTWPythonAdapter::fft)
    ;
}
