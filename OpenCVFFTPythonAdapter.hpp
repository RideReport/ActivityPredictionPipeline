#include <boost/python.hpp>

namespace py = boost::python;

class OpenCVFFTPythonAdapter {
public:
    OpenCVFFTPythonAdapter (int sampleSize);
    ~OpenCVFFTPythonAdapter ();

    py::list fft(py::list input);

protected:
    void* _fft;
    int _sampleSize;
};
