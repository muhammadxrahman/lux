# Start from a slim Python on Linux. This is NOT Apple Silicon;
# MLX here will use its CPU backend.
FROM python:3.14-slim

# A C++ compiler to build the sampler inside the image too.
RUN apt-get update && apt-get install -y g++ && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install the CPU build of MLX plus what the smoke test needs.
# mlx[cpu] is the Linux CPU backend; mlx-lm rides on top of it.
RUN pip install --no-cache-dir "mlx[cpu]" mlx-lm numpy pybind11

# Copy only what the smoke test needs for now.
COPY src/ ./src/
COPY proofs/docker_smoke.py ./proofs/docker_smoke.py

# Build the C++ sampler the cross-platform way (no Mac-only flag).
RUN c++ -O3 -Wall -shared -std=c++17 -fPIC \
    $(python -m pybind11 --includes) \
    src/sampler.cpp src/bindings.cpp \
    -o csampler$(python3-config --extension-suffix)

# Default command: run the smoke test.
CMD ["python", "proofs/docker_smoke.py"]