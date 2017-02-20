#include <boost/python.hpp>

namespace py = boost::python;

class UtilityAdapter {
public:
    UtilityAdapter();
    ~UtilityAdapter();

    py::object interpolateLinearRegular(py::list& inputX, py::list& inputY, float newSpacing, int outputLength, float initialOffset = 0.0);
    py::object interpolateSplineRegular(py::list& inputX, py::list& inputY, float newSpacing, int outputLength, float initialOffset = 0.0);

protected:
};
