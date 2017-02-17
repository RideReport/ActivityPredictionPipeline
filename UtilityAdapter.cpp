#include "UtilityAdapter.hpp"
#include "util.hpp"
#include "ActivityPredictor/Utility.c"
#include <vector>
using namespace std;

UtilityAdapter::UtilityAdapter() {
}

UtilityAdapter::~UtilityAdapter() {
}

py::object UtilityAdapter::interpolateRegular(
        py::list& inputX, py::list& inputY,
        float newSpacing, int outputLength)
{

    if (py::len(inputX) != py::len(inputY)) {
        throw length_error("Cannot interpolate X and Y different lengths");
    }
    auto inputXVec = vectorFromList<float>(inputX);
    auto inputYVec = vectorFromList<float>(inputY);
    auto outputVec = vector<float>(outputLength, 0.0);

    bool successful = ::interpolateRegular(
        inputXVec.data(),
        inputYVec.data(),
        inputXVec.size(),
        outputVec.data(),
        outputLength,
        newSpacing);

    if (successful) {
        return listFromVector(outputVec);
    }
    return py::object(); // return None
}

BOOST_PYTHON_MODULE(utilityadapter)
{
    py::class_<UtilityAdapter>("UtilityAdapter", py::init<>())
        .def("interpolateRegular", &UtilityAdapter::interpolateRegular)
    ;
}
