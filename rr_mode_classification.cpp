#include <vector>
#include <string>
#include <iostream>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "ActivityPredictor/RandomForestManager.h"
#include "util.hpp"

#ifdef __APPLE__
#include "AppleFFTPythonAdapter.hpp"
#endif
#include "FFTWPythonAdapter.hpp"

using namespace boost::python;
using namespace std;


// TODO: try vector_indexing_suite

class RandomForest {
public:
    RandomForest(int sampleSize, int samplingRateHz, std::string pathToModelFile) {
        _sampleSize = sampleSize;
        try {
            _manager = createRandomForestManager(sampleSize, samplingRateHz, pathToModelFile.c_str());
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

    list predict_proba(list& norms, list& norms2) {
        _checkClassCount();
        _checkNorms(norms);
        _checkNorms(norms2);

        auto normsVec = vectorFromList<float>(norms);
        auto normsVec2 = vectorFromList<float>(norms2);

        auto confidences = vector<float>(_n_classes);
        randomForestClassificationConfidences( _manager, normsVec.data(), normsVec2.data(), confidences.data(), _n_classes);

        list ret;
        for (float value : confidences) {
            ret.append(value);
        }
        return ret;
    }

    list prepareFeatures(list& norms, list& norms2) {
        _checkNorms(norms);
        _checkNorms(norms2);

        auto normsVec = vectorFromList<float>(norms);
        auto normsVec2 = vectorFromList<float>(norms2);

        auto featuresVec = vector<float>(RANDOM_FOREST_VECTOR_SIZE, 0.0);

        prepFeatureVector(_manager, featuresVec.data(), normsVec.data(), normsVec2.data());

        list ret;
        for (float value : featuresVec) {
            ret.append(value);
        }
        return ret;
    }

    int getFeatureCount() {
        return RANDOM_FOREST_VECTOR_SIZE;
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


BOOST_PYTHON_MODULE(rr_mode_classification)
{
    class_<RandomForest>("RandomForest", init<int, int, std::string>())
        .def("prepareFeatures", &RandomForest::prepareFeatures)
        .def("predict_proba", &RandomForest::predict_proba)
        .def("classLabels", &RandomForest::classLabels)
        .def("getFeatureCount", &RandomForest::getFeatureCount)
        .add_property("feature_count", &RandomForest::getFeatureCount)
    ;
}
