#include <vector>
#include <string>
#include <iostream>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "RandomForestManager.h"
using namespace boost::python;
using namespace std;

template<typename T>
vector<T> vectorFromList(list& l) {
    stl_input_iterator<T> begin(l), end;
    auto vec = vector<T>(begin, end);
    return vec;
}

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
        //if (_manager != NULL) {
        //    cout << "Trying to get class count on manager " << _manager << endl;
        //    _n_classes = randomForestGetClassCount(_manager);
        //}
    }
    ~RandomForest() {
        deleteRandomForestManager(_manager);
    }

    list predict_proba(list& norms) {
        const int N_CLASSES = 4;
        _checkNorms(norms);

        auto normsVec = vectorFromList<float>(norms);

        auto confidences = vector<float>(N_CLASSES);
        randomForestClassificationConfidences( _manager, normsVec.data(), confidences.data(), N_CLASSES);

        list ret;
        for (float value : confidences) {
            ret.append(value);
        }
        return ret;
    }

    list prepareFeatures(list& norms) {
        _checkNorms(norms);

        auto normsVec = vectorFromList<float>(norms);

        auto featuresVec = vector<float>(RANDOM_FOREST_VECTOR_SIZE, 0.0);

        prepFeatureVector(_manager, featuresVec.data(), normsVec.data());

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
        auto labelsVec = vector<int>(_n_classes, 0);
        randomForestGetClassLabels(_manager, labelsVec.data(), _n_classes);
        list ret;
        for (float value: labelsVec) {
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
};


BOOST_PYTHON_MODULE(rr_mode_classification)
{
    class_<RandomForest>("RandomForest", init<int, std::string>())
        .def("prepareFeatures", &RandomForest::prepareFeatures)
        .def("predict_proba", &RandomForest::predict_proba)
        .def("classLabels", &RandomForest::classLabels)
        .def("getFeatureCount", &RandomForest::getFeatureCount)
        .add_property("feature_count", &RandomForest::getFeatureCount)
    ;
}

