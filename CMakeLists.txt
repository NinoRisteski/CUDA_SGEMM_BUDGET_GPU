cmake_minimum_required(VERSION 3.18)

project(sgemm_cuda LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Prefer native GPU architecture when available (falls back to NVCC default).
if (NOT CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES native)
endif()

add_executable(sgemm
  src/main.cu
  src/naive.cu
)

target_compile_features(sgemm PRIVATE cxx_std_14)

# Enable separate compilation in case kernels grow.
set_target_properties(sgemm PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
)
