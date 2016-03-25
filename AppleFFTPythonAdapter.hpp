#include <boost/python.hpp>

namespace py = boost::python;

class AppleFFTPythonAdapter {
public:
    AppleFFTPythonAdapter(int sampleSize);
    ~AppleFFTPythonAdapter();

    py::list fft(py::list input);
    
protected:
    void* _fft;    
    int _sampleSize;
};
