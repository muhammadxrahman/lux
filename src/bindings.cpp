#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "sampler.h"

namespace py = pybind11;

// Zero-copy entry point. Reads logits straight out of a NumPy buffer instead
// of a Python list, so we never materialize ~128k PyFloat objects per token.
// forcecast lets callers pass float32 logits (what MLX produces) and we cast
// to double once, in C, rather than element by element in Python.
static py::array_t<double> sample_logits_np(
    py::array_t<double, py::array::c_style | py::array::forcecast> logits,
    double temperature,
    int top_k,
    double top_p) {
    auto buf = logits.request();
    const double* ptr = static_cast<const double*>(buf.ptr);

    // One contiguous memcpy-style fill, no per-element Python conversion.
    std::vector<double> in(ptr, ptr + buf.size);

    // Reuse the exact kernel that check_sampler.py verifies against the
    // NumPy reference -- the math is unchanged, only the marshalling differs.
    std::vector<double> out =
        sample_logits(std::move(in), temperature, top_k, top_p);

    return py::array_t<double>(
        static_cast<py::ssize_t>(out.size()), out.data());
}

PYBIND11_MODULE(csampler, m) {
    m.doc() = "C++ token sampler";

    // Original list-based entry point. Kept for the reference check and any
    // pure-Python caller; sample_logits_np is the one used in the hot path.
    m.def("sample_logits", &sample_logits,
          "Apply temperature, top-k, top-p; return probabilities (list in/out)",
          py::arg("logits"),
          py::arg("temperature") = 1.0,
          py::arg("top_k") = 0,
          py::arg("top_p") = 1.0);

    m.def("sample_logits_np", &sample_logits_np,
          "Same math as sample_logits, but zero-copy via the NumPy buffer",
          py::arg("logits"),
          py::arg("temperature") = 1.0,
          py::arg("top_k") = 0,
          py::arg("top_p") = 1.0);
}
