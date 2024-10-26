#include <iostream>
#include <vector>
#include <dnnl.hpp>
#include <cstdlib> // For aligned_alloc, free, and rand
#include <ctime>   // For seeding rand with time
#include <chrono>  // For measuring time
#include <cmath>
#include <cstdlib>
using namespace dnnl;

void batch_gemm_qk_with_onednn(float* Q_data, float* K_data, float* C_data,
                               int m, int head_size, int n1, int N, int lda, int ldb, int ldc, int batch_size) {
    // Create an engine and a stream
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Dimensions for Q, K, and output (C) matrices
    memory::dims q_dims = {batch_size, m, head_size};
    memory::dims k_dims = {batch_size, head_size, n1}; // Subset of n1 columns from the original N tokens
    memory::dims c_dims = {batch_size, m, n1};

    // Define the strides manually for Q, K, and output (C)
    memory::dims q_strides = {m * lda, lda, 1};           // Strides for Q
    memory::dims k_strides = {N * head_size, 1, head_size}; // Strides for K considering upfront allocation for N tokens
    memory::dims c_strides = {m * ldc, ldc, 1};           // Strides for output (C)

    // Create memory descriptors using the strides
    memory::desc q_md = memory::desc(q_dims, memory::data_type::f32, q_strides);
    memory::desc k_md = memory::desc(k_dims, memory::data_type::f32, k_strides);
    memory::desc c_md = memory::desc(c_dims, memory::data_type::f32, c_strides);

    // Create a matmul primitive descriptor
    matmul::primitive_desc matmul_pd(eng, q_md, k_md, c_md);

    // Create a matmul primitive
    matmul matmul_prim(matmul_pd);

    // Create memory objects for Q, K, and C
    auto Q_mem = memory(q_md, eng, Q_data);
    auto K_mem = memory(k_md, eng, K_data);
    auto C_mem = memory(c_md, eng, C_data);

    // Execute the matmul operation
    matmul_prim.execute(s, {
        {DNNL_ARG_SRC, Q_mem},
        {DNNL_ARG_WEIGHTS, K_mem},
        {DNNL_ARG_DST, C_mem}
    });

    // Wait for the computation to finish
    s.wait();
}

void batch_gemm_qk_full_onednn(float* Q_data, float* K_data, float* C_data,
                               int m, int head_size, int N, int lda, int ldb, int ldc, int batch_size) {
    // Create an engine and a stream
    engine eng(engine::kind::cpu, 0);
    stream s(eng);

    // Dimensions for Q, K, and output (C) matrices
    memory::dims q_dims = {batch_size, m, head_size};
    memory::dims k_dims = {batch_size, head_size, N}; // Full allocation for N columns
    memory::dims c_dims = {batch_size, m, N};

    // Define the strides manually for Q, K, and output (C)
    memory::dims q_strides = {m * lda, lda, 1};           // Strides for Q
    memory::dims k_strides = {N * head_size, 1, head_size}; // Strides for K considering full allocation for N tokens
    memory::dims c_strides = {m * ldc, ldc, 1};           // Strides for output (C)

    // Create memory descriptors using the strides
    memory::desc q_md = memory::desc(q_dims, memory::data_type::f32, q_strides);
    memory::desc k_md = memory::desc(k_dims, memory::data_type::f32, k_strides);
    memory::desc c_md = memory::desc(c_dims, memory::data_type::f32, c_strides);

    // Create a matmul primitive descriptor
    matmul::primitive_desc matmul_pd(eng, q_md, k_md, c_md);

    // Create a matmul primitive
    matmul matmul_prim(matmul_pd);

    // Create memory objects for Q, K, and C
    auto Q_mem = memory(q_md, eng, Q_data);
    auto K_mem = memory(k_md, eng, K_data);
    auto C_mem = memory(c_md, eng, C_data);

    // Execute the matmul operation
    matmul_prim.execute(s, {
        {DNNL_ARG_SRC, Q_mem},
        {DNNL_ARG_WEIGHTS, K_mem},
        {DNNL_ARG_DST, C_mem}
    });

    // Wait for the computation to finish
    s.wait();
}

int main(int argc, char *argv[]) {
    // Seed the random number generator
    std::srand(std::time(nullptr));

    // Example dimensions for Q, K, and output (C)
    int m = 1;                  // Number of query vectors (number of tokens being processed)
    int embedding_dim = 2048;   // Total embedding dimension for the model (e.g., 2048)
    int num_heads = atoi(argv[1]);         // Number of attention heads
    int head_size = embedding_dim / num_heads; // Calculate head_size (2048 / 32 = 64)
    int N = 2048;                 // Maximum number of key vectors (total sequence length)
    int batch_size = 128;         // Batch size

    // Set leading dimensions based on matrix dimensions
    int lda = head_size;        // Leading dimension for Q
    int ldb = head_size;        // Leading dimension for K
    int ldc = N;                // Leading dimension for output C
    int block_size = 1024;
    // Allocate memory for Q, K, and C using aligned_alloc
    float* Q_data = static_cast<float*>(aligned_alloc(64, batch_size * num_heads * m * lda * sizeof(float)));
    float* K_data = static_cast<float*>(aligned_alloc(64, batch_size * num_heads * N * ldb * sizeof(float))); // Upfront allocation for N tokens
    float* C_data = static_cast<float*>(aligned_alloc(64, batch_size * num_heads * m * ldc * sizeof(float)));
    float* C_data_full = static_cast<float*>(aligned_alloc(64, batch_size * num_heads * m * ldc * sizeof(float)));

    if (!Q_data || !K_data || !C_data || !C_data_full) {
        std::cerr << "Memory allocation failed." << std::endl;
        return -1;
    }

    // Initialize matrices Q and K with random values between 0 and 1
    for (int i = 0; i < batch_size * num_heads * m * lda; ++i) {
        Q_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }
    for (int i = 0; i < batch_size * num_heads * N * ldb; ++i) {
        K_data[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Initialize C_data and C_data_full to zero
    std::fill(C_data, C_data + batch_size * num_heads * m * ldc, 0.0f);
    std::fill(C_data_full, C_data_full + batch_size * num_heads * m * ldc, 0.0f);

    double total_time_strided = 0.0;
    for (int n1 = 1 ; n1 <= N; ++n1) {
        auto start = std::chrono::high_resolution_clock::now();
        batch_gemm_qk_with_onednn(Q_data, K_data, C_data, m, head_size, n1, N, lda, ldb, ldc, batch_size);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_subset = end - start;
        total_time_strided += elapsed_subset.count();
//std::cout << "Time for BMM Q . K^T (subset n1 = " << n1 << "): " << elapsed_subset.count() << " seconds" << std::endl;
    }

    // Measure the total time for the full BMM operation run the same number of times
    double total_time_full = 0.0;
    for (int i = 1; i <= N; ++i) {
        auto start_full = std::chrono::high_resolution_clock::now();
        batch_gemm_qk_full_onednn(Q_data, K_data, C_data_full, m, head_size, N, lda, ldb, ldc, batch_size);
        auto end_full = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_full = end_full - start_full;
        total_time_full += elapsed_full.count();
    }

    // Calculate the speedup
    double speedup = total_time_full / total_time_strided;
    std::cout << "Total time for strided BMM operations: " << total_time_strided << " seconds" << std::endl;
    std::cout << "Total time for full BMM operations (repeated " << N << " times): " << total_time_full << " seconds"
            << std::endl;
    std::cout << "Speedup of full BMM over strided BMM: " << total_time_strided/total_time_full << " times" << "\n";
    // Free the allocated memory
    std::cout << "\n";
    free(Q_data);
    free(K_data);
    free(C_data);
    free(C_data_full);

    return 0;
}
