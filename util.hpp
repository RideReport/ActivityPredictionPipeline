#ifndef UTIL_HPP
#define UTIL_HPP

#include <boost/python.hpp>
#include <boost/python/stl_iterator.hpp>

template<typename T>
std::vector<T> vectorFromList(boost::python::list& l) {
    boost::python::stl_input_iterator<T> begin(l), end;
    auto vec = std::vector<T>(begin, end);
    return vec;
}

#endif
