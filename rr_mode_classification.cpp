#include <vector>
#include <string>
#include <iostream>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "ActivityPredictor/RandomForestManager.h"
#include "util.hpp"

namespace py = boost::python;
using namespace std;

#ifndef PYTHON_MODULE_NAME
#define PYTHON_MODULE_NAME rr_mode_classification
#endif

// TODO: try vector_indexing_suite

class RandomForest {
public:
    RandomForest(int sampleSize, int samplingRateHz, py::object pathToModelFile) {
        _sampleSize = sampleSize;
        py::extract<char const*> modelPath(pathToModelFile);
        try {
            _manager = createRandomForestManager(sampleSize, samplingRateHz, modelPath.check() ? modelPath() : NULL);
        }
        catch (std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "Unknown error");
        }
        _n_classes = -1;
    }
    ~RandomForest() {
        deleteRandomForestManager(_manager);
    }

    py::list classifyMagnitudes(py::list& norms) {
        _checkCanPredict();
        _checkClassCount();
        _checkNorms(norms);
        auto normsVec = vectorFromList<float>(norms);

        auto confidences = vector<float>(_n_classes);

        randomForestClassifyMagnitudeVector( _manager, normsVec.data(), confidences.data(), _n_classes);

        return listFromVector(confidences);
    }

    py::list classifyFeatures(py::list& features) {
        _checkCanPredict();
        _checkClassCount();
        _checkFeatures(features);
        auto featuresVec = vectorFromList<float>(features);
        auto confidences = vector<float>(_n_classes);

        randomForestClassifyFeatures(_manager, featuresVec.data(), confidences.data(), _n_classes);

        return listFromVector(confidences);
    }

    py::list classifySignal(py::list& readingsList) {
        _checkCanPredict();
        _checkClassCount();
        auto readings = readingsFromList(readingsList);
        auto confidences = vector<float>(_n_classes);
        bool successful = randomForestClassifyAccelerometerSignal(_manager, readings.data(), readings.size(), confidences.data(), _n_classes);

        if (!successful) {
            throw std::runtime_error("Failed to classify signal; probably not enough data");
        }

        return listFromVector(confidences);
    }

    py::list prepareFeaturesFromSignal(py::list& readingsList, float offsetSeconds = 0.f) {
        auto readings = readingsFromList(readingsList);
        auto features = vector<float>(RANDOM_FOREST_VECTOR_SIZE);
        bool successful = randomForestPrepareFeaturesFromAccelerometerSignal(_manager, readings.data(), readings.size(), features.data(), features.size(), offsetSeconds);

        if (!successful) {
            throw std::runtime_error("Failed to prepare features; probably not enough data");
        }

        return listFromVector(features);
    }

    py::list prepareFeatures(py::list& norms) {
        _checkNorms(norms);
        auto normsVec = vectorFromList<float>(norms);
        auto featuresVec = vector<float>(getFeatureCount(), 0.0);

        prepFeatureVector(_manager, featuresVec.data(), normsVec.data());

        return listFromVector(featuresVec);
    }

    int getFeatureCount() {
        return RANDOM_FOREST_VECTOR_SIZE;
    }

    py::list classLabels() {
        _checkCanPredict();
        _checkClassCount();
        auto labelsVec = vector<int>(_n_classes, 0);
        randomForestGetClassLabels(_manager, labelsVec.data(), _n_classes);

        return listFromVector(labelsVec);
    }


protected:
    RandomForestManager* _manager;
    int _sampleSize;
    int _n_classes;
    void _checkNorms(py::list& norms) {
        if (py::len(norms) != _sampleSize) {
            throw std::length_error("Cannot classify vector with length that does not match sample size");
        }
    }

    void _checkFeatures(py::list& features) {
        if (py::len(features) != RANDOM_FOREST_VECTOR_SIZE) {
            throw std::length_error("Cannot classify features with length that does not match expected feature count");
        }
    }

    void _checkClassCount() {
        if (_n_classes == -1) {
            _n_classes = randomForestGetClassCount(_manager);
        }
    }

    void _checkCanPredict() {
        if (!randomForestManagerCanPredict(_manager)) {
            throw std::runtime_error("RF Manager cannot predict");
        }
    }

    vector<AccelerometerReading> readingsFromList(py::list& readingsList) {
        int listLength = py::len(readingsList);
        auto readings = vector<AccelerometerReading>(listLength);
        for (int i = 0; i < listLength; ++i) {
            py::dict readingDict = py::extract<py::dict>(readingsList[i]);
            readings[i].x = py::extract<float>(readingDict['x']);
            readings[i].y = py::extract<float>(readingDict['y']);
            readings[i].z = py::extract<float>(readingDict['z']);
            readings[i].t = py::extract<float>(readingDict['t']);
            // cerr << "reading i=" << i << " " << readings[i].x << " " << readings[i].y << " " << readings[i].z << " t=" << readings[i].t << endl;
        }
        return readings;
    }
};

BOOST_PYTHON_MEMBER_FUNCTION_OVERLOADS(RandomForest_prepareFeaturesFromSignal_overloads, RandomForest::prepareFeaturesFromSignal, 1, 2);

BOOST_PYTHON_MODULE(PYTHON_MODULE_NAME)
{
    py::class_<RandomForest>("RandomForest", py::init<int, int, py::object>())
        .def("prepareFeatures", &RandomForest::prepareFeatures)
        .def("classifyFeatures", &RandomForest::classifyFeatures)
        .def("classifyMagnitudes", &RandomForest::classifyMagnitudes)
        .def("classifySignal", &RandomForest::classifySignal)
        .def("prepareFeaturesFromSignal", &RandomForest::prepareFeaturesFromSignal,
            RandomForest_prepareFeaturesFromSignal_overloads(
                py::args("offsetSeconds"), ""
            ))
        .def("classLabels", &RandomForest::classLabels)
        .def("getFeatureCount", &RandomForest::getFeatureCount)
        .add_property("feature_count", &RandomForest::getFeatureCount)
    ;
}
