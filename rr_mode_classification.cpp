#include <vector>
#include <string>
#include <iostream>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "RandomForestManager.h"
using namespace boost::python;
using namespace std;


// TODO: try vector_indexing_suite

class RandomForest {
public:
    RandomForest(int sampleSize, std::string pathToModelFile) {
        _sampleSize = sampleSize;
        try {
            _manager = createRandomForestManager(sampleSize, pathToModelFile.c_str());
        }
        catch (std::exception& e) {
            PyErr_SetString(PyExc_RuntimeError, e.what());
        }
        catch (...) {
            PyErr_SetString(PyExc_RuntimeError, "Unknown error");
        }
    }
    ~RandomForest() {
        deleteRandomForestManager(_manager);
    }
    int classify(boost::python::list& norms) {
        _checkNorms(norms);

        // Convert from list to vector
        stl_input_iterator<float> begin(norms), end;
        auto normsVector = vector<float>(begin, end);
        return randomForesetClassifyMagnitudeVector(_manager, normsVector.data());
    }

    list predict_proba(list& norms) {
        const int N_CLASSES = 4;
        _checkNorms(norms);
        stl_input_iterator<float> begin(norms), end;
        auto normsVec = vector<float>(begin, end);
        auto confidences = vector<float>(N_CLASSES);
        randomForestClassificationConfidences(_manager, normsVec.data(), confidences.data(), N_CLASSES);

        list ret;
        for (float value : confidences) {
            ret.append(value);
        }
        return ret;
    }

    list prepareFeatures(list& norms) {
        _checkNorms(norms);

        stl_input_iterator<float> begin(norms), end;
        auto normsVec = vector<float>(begin, end);

        auto featuresVec = vector<float>(RANDOM_FOREST_VECTOR_SIZE, 0.0);

        prepFeatureVector(_manager, featuresVec.data(), normsVec.data());

        list ret;
        for (float value : featuresVec) {
            ret.append(value);
        }
        return ret;
    }


protected:
    RandomForestManager* _manager;
    int _sampleSize;

    void _checkNorms(list& norms) {
        if (len(norms) != _sampleSize) {
            throw std::length_error("Cannot classify vector with length that does not match sample size");
        }
    }
};


BOOST_PYTHON_MODULE(rr_mode_classification)
{
    class_<RandomForest>("RandomForest", init<int, std::string>())
        .def("classify", &RandomForest::classify)
        .def("prepareFeatures", &RandomForest::prepareFeatures)
        .def("predict_proba", &RandomForest::predict_proba)
    ;
}

