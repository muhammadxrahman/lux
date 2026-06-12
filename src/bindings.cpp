#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "sampler.h"

namespace py = pybind11;

PYBIND11_MODULE(csampler, m) {
    m.doc() = "C++ token sampler";
    m.def("sample_logits", &sample_logits,
          "Apply temperature, top-k, top-p; return probabilities",
          py::arg("logits"),
          py::arg("temperature") = 1.0,
          py::arg("top_k") = 0,
          py::arg("top_p") = 1.0);
}