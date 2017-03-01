#include "UtilityAdapter.hpp"
#include "util.hpp"
#include "ActivityPredictor/Utility.cpp"
#include <vector>
using namespace std;

UtilityAdapter::UtilityAdapter() {
}

UtilityAdapter::~UtilityAdapter() {
}

py::object UtilityAdapter::interpolateSplineRegular(
        py::list& inputX, py::list& inputY,
        float newSpacing, int outputLength, float initialOffset)
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
        newSpacing,
        initialOffset);

    if (!successful) {
        throw range_error("Insufficient data to interpolate up to desired length");
    }
    return listFromVector(outputVec);
}

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(UtilityAdapter_interpolateSplineRegular_overloads, UtilityAdapter::interpolateSplineRegular, 4, 5);

BOOST_PYTHON_MODULE(utilityadapter)
{
    py::class_<UtilityAdapter>("UtilityAdapter", py::init<>())
        .def("interpolateSplineRegular", &UtilityAdapter::interpolateSplineRegular,
            UtilityAdapter_interpolateSplineRegular_overloads(
                py::args("initialOffset"), ""
            ))
    ;
}
