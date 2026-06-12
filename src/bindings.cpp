#include <pybind11/pybind11.h>
#include <pybind11/stl.h>   // lets pybind auto-convert Python lists <-> std::vector
#include "sampler.h"

namespace py = pybind11;

// This macro creates the importable Python module.
// The name "csampler" here MUST match the module name in CMakeLists.txt.
PYBIND11_MODULE(csampler, m) {
    m.doc() = "C++ token sampler";
    m.def("sample_logits", &sample_logits,
          "Apply temperature and return probabilities",
          py::arg("logits"), py::arg("temperature") = 1.0);
}