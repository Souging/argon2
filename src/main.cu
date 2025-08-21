#include <cuda_runtime.h>
#include <openssl/evp.h>
#include <iostream>
#include <sstream>
#include <iomanip>
#include <cstring>
#include <string>
#include <cstdlib>
#include <ctime>

#include "../include/argon2.cuh"

#define BATCH_SIZE     2048
#define MAX_PASS_LEN   256
#define HASH_LEN       32

std::string to_hex(const unsigned char* data, size_t len) {
    std::stringstream ss;
    ss << std::hex << std::setfill('0');
    for (size_t i = 0; i < len; ++i) {
        ss << std::setw(2) << static_cast<unsigned>(data[i]);
    }
    return ss.str();
}

int count_leading_zeros_bitwise(const unsigned char* hash) {
    int zeros = 0;
    for (int i = 0; i < 32; ++i) {
        unsigned char b = hash[i];
        if (b == 0) {
            zeros += 8;
        } else {
            for (int j = 7; j >= 0; --j) {
                if ((b >> j) & 1)
                    return zeros;
                zeros++;
            }
        }
    }
    return zeros;
}

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <blockData> <difficulty>\n";
        return -1;
    }

    const std::string blockData = argv[1];
    const int difficulty = std::stoi(argv[2]);

    srand((unsigned)time(0));
    long long nonceOne = ((long long)rand() * RAND_MAX + rand()) % 1000000000000LL;

    std::string prefix = blockData + std::to_string(nonceOne) + ":";
    const std::string salt = "tabcoin-v1";

    // Initialize CUDA stream
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    // Managed memory buffers
    uint8_t *d_passes, *d_salt, *d_hashes;
    cudaMallocManaged(&d_passes, BATCH_SIZE * MAX_PASS_LEN);
    cudaMallocManaged(&d_salt, 32);
    cudaMallocManaged(&d_hashes, BATCH_SIZE * HASH_LEN);

    // Copy salt to device memory
    memcpy(d_salt, salt.c_str(), salt.length());
    d_salt[salt.length()] = '\0';

    unsigned long long nonceTwo = 0;

    while (true) {
        // Prepare batch of passwords
        for (int i = 0; i < BATCH_SIZE; ++i) {
            std::string fullPass = prefix + std::to_string(nonceTwo + i);
            size_t pass_len = fullPass.length();
            if (pass_len >= MAX_PASS_LEN) pass_len = MAX_PASS_LEN - 1;
            memcpy(d_passes + i * MAX_PASS_LEN, fullPass.c_str(), pass_len);
            d_passes[i * MAX_PASS_LEN + pass_len] = '\0';
        }

        // Launch kernel
        dim3 grid((BATCH_SIZE + 255) / 256); // Block size 设为 256 threads
        dim3 block(256);
        argon2_kernel_batch<<<grid, block, 0, stream>>>(
            d_passes, MAX_PASS_LEN,
            d_salt, salt.length(),
            d_hashes, BATCH_SIZE
        );

        cudaStreamSynchronize(stream);

        // Check results
        for (int i = 0; i < BATCH_SIZE; ++i) {
            EVP_MD_CTX* mdctx = EVP_MD_CTX_new();
            EVP_DigestInit_ex(mdctx, EVP_sha256(), NULL);
            EVP_DigestUpdate(mdctx, d_hashes + i * HASH_LEN, HASH_LEN);

            unsigned char sha256[EVP_MAX_MD_SIZE];
            unsigned int digest_len;
            EVP_DigestFinal_ex(mdctx, sha256, &digest_len);
            EVP_MD_CTX_free(mdctx);

            int leading_bits = count_leading_zeros_bitwise(sha256);
            if (leading_bits >= difficulty) {
                std::cout << "Found solution!" << std::endl;
                std::cout << "nonceOne: " << nonceOne << std::endl;
                std::cout << "nonceTwo: " << (nonceTwo + i) << std::endl;
                std::cout << "shaHash:  " << to_hex(sha256, digest_len) << std::endl;
                std::cout << "blockData:" << blockData << std::endl;
                std::cout << "hashResult:" << to_hex(d_hashes + i * HASH_LEN, HASH_LEN) << std::endl;
                cudaFree(d_passes);
                cudaFree(d_salt);
                cudaFree(d_hashes);
                cudaStreamDestroy(stream);
                return 0;
            }
        }

        nonceTwo += BATCH_SIZE;
        if ((nonceTwo / BATCH_SIZE) % 1000 == 0) {
            std::cout << "[CUDA Miner] Tried up to nonceTwo=" << nonceTwo << std::endl;
        }
    }

    // Cleanup
    cudaFree(d_passes);
    cudaFree(d_salt);
    cudaFree(d_hashes);
    cudaStreamDestroy(stream);

    return 0;
}
