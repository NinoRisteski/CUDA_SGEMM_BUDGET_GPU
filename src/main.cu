#include <cuda_runtime.h>

#include <cstdlib>
#include <iostream>
#include <random>
#include <vector>

__global__ void sgemm_naive(int M, int N, int K, float alpha, const float *A,
                            const float *B, float beta, float *C);

static void checkCuda(cudaError_t err, const char *msg) {
  if (err != cudaSuccess) {
    std::cerr << msg << ": " << cudaGetErrorString(err) << std::endl;
    std::exit(EXIT_FAILURE);
  }
}

int main(int argc, char **argv) {
  if (argc < 2) {
    std::cerr << "Usage: DEVICE=<device_id> " << argv[0] << " <kernel_id> "
              << "[M N K]" << std::endl;
    std::cerr << "Available kernels: 0 = naive" << std::endl;
    return EXIT_FAILURE;
  }

  const int kernel_id = std::atoi(argv[1]);
  if (kernel_id != 0) {
    std::cerr << "Unsupported kernel_id " << kernel_id << " (only 0 = naive)"
              << std::endl;
    return EXIT_FAILURE;
  }

  int M = 512, N = 512, K = 512;
  if (argc == 5) {
    M = std::atoi(argv[2]);
    N = std::atoi(argv[3]);
    K = std::atoi(argv[4]);
  }

  const char *dev_env = std::getenv("DEVICE");
  int device_id = dev_env ? std::atoi(dev_env) : 0;
  checkCuda(cudaSetDevice(device_id), "cudaSetDevice failed");

  const size_t size_A = static_cast<size_t>(M) * K;
  const size_t size_B = static_cast<size_t>(K) * N;
  const size_t size_C = static_cast<size_t>(M) * N;

  std::vector<float> h_A(size_A);
  std::vector<float> h_B(size_B);
  std::vector<float> h_C(size_C, 0.0f);

  std::mt19937 rng(42);
  std::uniform_real_distribution<float> dist(0.0f, 1.0f);
  for (auto &v : h_A) v = dist(rng);
  for (auto &v : h_B) v = dist(rng);

  float *d_A = nullptr;
  float *d_B = nullptr;
  float *d_C = nullptr;
  checkCuda(cudaMalloc(&d_A, size_A * sizeof(float)), "cudaMalloc d_A");
  checkCuda(cudaMalloc(&d_B, size_B * sizeof(float)), "cudaMalloc d_B");
  checkCuda(cudaMalloc(&d_C, size_C * sizeof(float)), "cudaMalloc d_C");

  checkCuda(cudaMemcpy(d_A, h_A.data(), size_A * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy A");
  checkCuda(cudaMemcpy(d_B, h_B.data(), size_B * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy B");
  checkCuda(cudaMemcpy(d_C, h_C.data(), size_C * sizeof(float),
                       cudaMemcpyHostToDevice),
            "cudaMemcpy C");

  const dim3 block(16, 16);
  const dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  const float alpha = 1.0f;
  const float beta = 0.0f;

  sgemm_naive<<<grid, block>>>(M, N, K, alpha, d_A, d_B, beta, d_C);
  checkCuda(cudaGetLastError(), "Kernel launch failed");
  checkCuda(cudaDeviceSynchronize(), "Kernel execution failed");

  checkCuda(cudaMemcpy(h_C.data(), d_C, size_C * sizeof(float),
                       cudaMemcpyDeviceToHost),
            "cudaMemcpy C back");

  double checksum = 0.0;
  for (float v : h_C) checksum += v;
  std::cout << "Kernel 0 (naive) done. C checksum: " << checksum << std::endl;

  cudaFree(d_A);
  cudaFree(d_B);
  cudaFree(d_C);
  return EXIT_SUCCESS;
}
