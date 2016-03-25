#include <boost/python.hpp>

namespace py = boost::python;

class FFTWPythonAdapter {
public:
    FFTWPythonAdapter(int sampleSize);
    ~FFTWPythonAdapter();

    py::list fft(py::list input);
    
protected:
    void* _fft;    
    int _sampleSize;
};
