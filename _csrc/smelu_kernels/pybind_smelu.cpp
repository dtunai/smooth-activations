#include "smelu.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

template <typename T>
void declare_SmeLU(py::module& m, const std::string& name) {
    py::class_<SmeLU<T>>(m, name.c_str())
            .def(py::init<T, std::size_t>(), py::arg("alpha_value") = T(1.0), py::arg("size") = std::size_t(1))
            .def("forward", &SmeLU<T>::forward)
            .def("backward", &SmeLU<T>::backward);
}

PYBIND11_MODULE(smelu, m) {
declare_SmeLU<float>(m, "SmeLUf");
declare_SmeLU<double>(m, "SmeLUD");
}
