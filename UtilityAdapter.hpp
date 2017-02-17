#include <boost/python.hpp>

namespace py = boost::python;

class UtilityAdapter {
public:
    UtilityAdapter();
    ~UtilityAdapter();

    py::object interpolateRegular(py::list& inputX, py::list& inputY, float newSpacing, int outputLength);

protected:
};
