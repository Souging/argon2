#ifndef ARGON2_CUH
#define ARGON2_CUH

#include <cuda_runtime.h>
#include <cstdint>

__global__ void argon2_kernel_batch(
    uint8_t* d_passes, int max_pass_len,
    uint8_t* d_salt, int salt_len,
    uint8_t* d_out, int batch_size
);

#endif
