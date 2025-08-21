#include "../include/argon2.cuh"
#include <cstring>
#include <cstdint>
#include <cuda_runtime.h>

#define ARGON2_BLOCK_SIZE   1024
#define ARGON2_QWORDS       (ARGON2_BLOCK_SIZE / 8)

// Device Helper Functions

__device__ inline uint64_t rotr64(uint64_t w, unsigned c) {
    return (w >> c) | (w << (64 - c));
}

__device__ void g(uint64_t *a, uint64_t *b, uint64_t *c, uint64_t *d) {
    *a += *b;
    *d ^= *a;
    *d = rotr64(*d, 32);

    *c += *d;
    *b ^= *c;
    *b = rotr64(*b, 24);

    *a += *b;
    *d ^= *a;
    *d = rotr64(*d, 16);

    *c += *d;
    *b ^= *c;
    *b = rotr64(*b, 63);
}

__device__ void init_blocks(uint64_t* memory, uint8_t* pass, int pass_len, uint8_t* salt, int salt_len) {
    for (int i = 0; i < ARGON2_QWORDS; ++i) {
        uint64_t val = 0;
        for (int j = 0; j < 8 && i * 8 + j < pass_len; ++j)
            val |= ((uint64_t)pass[i * 8 + j]) << (j * 8);
        for (int j = 0; j < 8 && i * 8 + j < salt_len; ++j)
            val ^= ((uint64_t)salt[i * 8 + j]) << (j * 8);
        memory[i] = val;
    }
}

__device__ void fill_memory(uint64_t* memory) {
    for (int t = 0; t < 3; ++t) {
        for (int slice = 0; slice < 4; ++slice) {
            for (int index = slice * 32; index < (slice + 1) * 32; ++index) {
                uint64_t ref_index = memory[index % ARGON2_QWORDS];
                ref_index %= ARGON2_QWORDS;

                uint64_t* prev = &memory[(index - 1 + ARGON2_QWORDS) % ARGON2_QWORDS];
                uint64_t* ref = &memory[ref_index];

                for (int i = 0; i < ARGON2_QWORDS; i += 4)
                    g(&memory[i], &memory[i+1], &memory[i+2], &memory[i+3]);

                memory[index] ^= *prev ^ *ref;
            }
        }
    }
}

// 支持批量执行的 CUDA Kernel
__global__ void argon2_kernel_batch(
        uint8_t* d_passes, int max_pass_len,
        uint8_t* d_salt, int salt_len,
        uint8_t* d_out, int batch_size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= batch_size) return;

    // 每个线程独立计算一个 Argon2 哈希
    uint64_t memory[ARGON2_QWORDS];  // 局部队列暂存（静态分配）

    uint8_t* pass_ptr = d_passes + tid * max_pass_len;
    int pass_len = 0;
    while (pass_len < max_pass_len && pass_ptr[pass_len] != '\0') pass_len++;

    init_blocks(memory, pass_ptr, pass_len, d_salt, salt_len);
    fill_memory(memory);

    // 提取前 32 bytes 作为输出
    for (int i = 0; i < 32; ++i) {
        d_out[tid * 32 + i] = ((uint8_t*)memory)[i];
    }
}
