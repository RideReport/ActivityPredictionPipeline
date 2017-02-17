#ifndef UTIL_HPP
#define UTIL_HPP

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>
#include <vector>

template<typename T>
std::vector<T> vectorFromList(boost::python::list& l) {
    boost::python::stl_input_iterator<T> begin(l), end;
    auto vec = std::vector<T>(begin, end);
    return vec;
}

template<typename T>
boost::python::list listFromVector(std::vector<T> vec) {
    boost::python::list ret;
    for (T value : vec) {
        ret.append(value);
    }
    return ret;
}

#endif
