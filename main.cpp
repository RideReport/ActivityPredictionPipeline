#include <vector>
#include <string>

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include "RandomForestManager.h"
using namespace boost::python;
using namespace std;

char const* greet(unsigned x)
{
    static char const* msgs[] = { "hello", "boost", "world" };
    if (x > 2) {
        throw std::range_error("greet: index out of range");
    }

    return msgs[x];
}

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
        if (len(norms) != _sampleSize) {
            throw std::length_error("Cannot classify vector with length that does not match sample size");
        }
        // Convert from list to vector
        stl_input_iterator<float> begin(norms), end;
        auto normsVector = vector<float>(begin, end);
        return randomForesetClassifyMagnitudeVector(_manager, normsVector.data());
    }

protected:
    RandomForestManager* _manager;
    int _sampleSize;
};


BOOST_PYTHON_MODULE(main)
{
    class_<RandomForest>("RandomForest", init<int, std::string>())
        .def("classify", &RandomForest::classify)
    ;
}

