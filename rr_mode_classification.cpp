#include <vector>
#include <string>
#include <iostream>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "ActivityPredictor/RandomForestManager.h"
#include "util.hpp"

using namespace boost::python;
using namespace std;

#ifndef PYTHON_MODULE_NAME
#define PYTHON_MODULE_NAME rr_mode_classification
#endif

// TODO: try vector_indexing_suite

class RandomForest {
public:
    RandomForest(int sampleSize, int samplingRateHz, std::string pathToModelFile, bool accelOnly) {
        _sampleSize = sampleSize;
        try {
            _manager = createRandomForestManager(sampleSize, samplingRateHz, pathToModelFile.c_str(), accelOnly);
        }
        catch (std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "Unknown error");
        }
        _n_classes = -1;
        _accelOnly = accelOnly;
    }
    ~RandomForest() {
        deleteRandomForestManager(_manager);
    }

    list predict_proba(list& norms, list& norms2) {
        _checkClassCount();
        _checkNorms(norms);
        auto normsVec = vectorFromList<float>(norms);

        auto confidences = vector<float>(_n_classes);

        if (_accelOnly) {
            randomForestClassificationConfidencesAccelerometerOnly(_manager, normsVec.data(), confidences.data(), _n_classes);
        }
        else {
            _checkNorms(norms2);
            auto normsVec2 = vectorFromList<float>(norms2);
            randomForestClassificationConfidences( _manager, normsVec.data(), normsVec2.data(), confidences.data(), _n_classes);
        }

        list ret;
        for (float value : confidences) {
            ret.append(value);
        }
        return ret;
    }

    list prepareFeatures(list& norms, list& norms2) {
        _checkNorms(norms);
        auto normsVec = vectorFromList<float>(norms);
        auto featuresVec = vector<float>(getFeatureCount(), 0.0);

        if (_accelOnly) {
            prepFeatureVectorAccelerometerOnly(_manager, featuresVec.data(), normsVec.data());
        }
        else {
            _checkNorms(norms2);
            auto normsVec2 = vectorFromList<float>(norms2);
            prepFeatureVector(_manager, featuresVec.data(), normsVec.data(), normsVec2.data());
        }

        list ret;
        for (float value : featuresVec) {
            ret.append(value);
        }
        return ret;
    }

    int getFeatureCount() {
        if (_accelOnly) {
            return RANDOM_FOREST_VECTOR_SIZE_ACCELEROMETER_ONLY;
        }
        else {
            return RANDOM_FOREST_VECTOR_SIZE;
        }
    }

    list classLabels() {
        _checkClassCount();
        auto labelsVec = vector<int>(_n_classes, 0);
        randomForestGetClassLabels(_manager, labelsVec.data(), _n_classes);
        list ret;
        for (int value: labelsVec) {
            ret.append(value);
        }
        return ret;
    }


protected:
    RandomForestManager* _manager;
    int _sampleSize;
    int _n_classes;
    bool _accelOnly;

    void _checkNorms(list& norms) {
        if (len(norms) != _sampleSize) {
            throw std::length_error("Cannot classify vector with length that does not match sample size");
        }
    }

    void _checkClassCount() {
        if (_n_classes == -1) {
            _n_classes = randomForestGetClassCount(_manager);
        }
    }

};


BOOST_PYTHON_MODULE(PYTHON_MODULE_NAME)
{
    class_<RandomForest>("RandomForest", init<int, int, std::string, bool>())
        .def("prepareFeatures", &RandomForest::prepareFeatures)
        .def("predict_proba", &RandomForest::predict_proba)
        .def("classLabels", &RandomForest::classLabels)
        .def("getFeatureCount", &RandomForest::getFeatureCount)
        .add_property("feature_count", &RandomForest::getFeatureCount)
    ;
}
