#include "UtilityAdapter.hpp"
#include "util.hpp"
#include "ActivityPredictor/Utility.cpp"
#include <vector>
using namespace std;

UtilityAdapter::UtilityAdapter() {
}

UtilityAdapter::~UtilityAdapter() {
}

py::object UtilityAdapter::interpolateLinearRegular(
        py::list& inputX, py::list& inputY,
        float newSpacing, int outputLength)
{
    if (py::len(inputX) != py::len(inputY)) {
        throw length_error("Cannot interpolate X and Y different lengths");
    }
    auto inputXVec = vectorFromList<float>(inputX);
    auto inputYVec = vectorFromList<float>(inputY);
    auto outputVec = vector<float>(outputLength, 0.0);

    bool successful = ::interpolateLinearRegular(
        inputXVec.data(),
        inputYVec.data(),
        inputXVec.size(),
        outputVec.data(),
        outputLength,
        newSpacing);

    if (!successful) {
        throw range_error("Insufficient data to interpolate up to desired length");
    }
    return listFromVector(outputVec);
}

py::object UtilityAdapter::interpolateSplineRegular(
        py::list& inputX, py::list& inputY,
        float newSpacing, int outputLength)
{
    if (py::len(inputX) != py::len(inputY)) {
        throw length_error("Cannot interpolate X and Y different lengths");
    }
    auto inputXVec = vectorFromList<float>(inputX);
    auto inputYVec = vectorFromList<float>(inputY);
    auto outputVec = vector<float>(outputLength, 0.0);

    bool successful = ::interpolateSplineRegular(
        inputXVec.data(),
        inputYVec.data(),
        inputXVec.size(),
        outputVec.data(),
        outputLength,
        newSpacing);

    if (!successful) {
        throw range_error("Insufficient data to interpolate up to desired length");
    }
    return listFromVector(outputVec);
}


BOOST_PYTHON_MODULE(utilityadapter)
{
    py::class_<UtilityAdapter>("UtilityAdapter", py::init<>())
        .def("interpolateLinearRegular", &UtilityAdapter::interpolateLinearRegular)
        .def("interpolateSplineRegular", &UtilityAdapter::interpolateSplineRegular)
    ;
}
