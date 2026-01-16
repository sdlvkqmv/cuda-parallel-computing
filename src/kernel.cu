#include <iostream>
#include <vector>
#include <cmath>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include "DS_definitions.h"
#include "DS_timer.h"
#include "Error.h"

// 1. Model Dimensions
#define N_SAMPLES 8192    // 시퀀스 길이 (N)
#define HEAD_DIM  32     // 헤드 차원 (D)

// 2. GPU Kernel Tuning
#define TILE_SIZE 32  // Flash Attention 타일 크기 (기존 16에서 32로 변경 권장)
#define NAIVE_BLOCK_SIZE 256 // Naive 커널의 Block Dim

// 3. Verification & Others
#define EPSILON 1e-3f     // 결과 검증 오차 범위
#define RANDOM_SEED 2026  // 재현성을 위한 시드값

// =========================================================
// 1. Naive Attention Kernel (메모리 비효율적 방식)
// 특징: N*N Score 행렬을 Global Memory에 쓰고 읽음
// =========================================================
__global__ void naive_attention_kernel(float* Q, float* K, float* V, float* O, float* Score, int N, int D) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) return;

    // [Step 1] Q * K^T 연산 -> Score 행렬(Global Memory)에 저장
    float max_val = -1e20f;
    for (int j = 0; j < N; j++) {
        float sum = 0.0f;
        for (int d = 0; d < D; d++) {
            sum += Q[row * D + d] * K[j * D + d];
        }
        // Score 저장 (Global Memory Write 발생!)
        float s = sum / sqrtf((float)D);
        Score[row * N + j] = s;
        if (s > max_val) max_val = s;
    }

    // [Step 2] Softmax 계산
    float sum_exp = 0.0f;
    for (int j = 0; j < N; j++) {
        // Score 읽기 (Global Memory Read!)
        float val = expf(Score[row * N + j] - max_val);
        Score[row * N + j] = val; // 확률값 덮어쓰기 (Global Write!)
        sum_exp += val;
    }

    // [Step 3] P * V 연산 -> Output
    for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int j = 0; j < N; j++) {
            // Score와 V 읽기 (Global Memory Read!)
            acc += (Score[row * N + j] / sum_exp) * V[j * D + d];
        }
        O[row * D + d] = acc;
    }
}

// =========================================================
// 2. Flash Attention Kernel (Online Softmax + Tiling)
// 특징: Score 행렬을 저장하지 않고 Shared Memory에서 해결
// =========================================================
__global__ void flash_attention_kernel(float* Q, float* K, float* V, float* O, int N, int D) {
    __shared__ float s_K[TILE_SIZE][HEAD_DIM];
    __shared__ float s_V[TILE_SIZE][HEAD_DIM];

    int row = blockIdx.x * blockDim.x + threadIdx.x;

    // 유효한 행 범위 체크
    if (row >= N) return;

    // Q는 레지스터에 캐싱
    float q_reg[128];
    for (int d = 0; d < D; d++) {
        q_reg[d] = Q[row * D + d];
    }

    // Online Softmax 변수
    float m = -1e20f; // max
    float l = 0.0f;   // sum
    float acc[128] = { 0.0f }; // output accumulator

    int num_tiles = (N + TILE_SIZE - 1) / TILE_SIZE;

    // 타일 루프
    for (int t = 0; t < num_tiles; t++) {
        int col = t * TILE_SIZE + threadIdx.x;

        // Load K, V to Shared Memory
        if (col < N) {
            for (int d = 0; d < D; d++) {
                s_K[threadIdx.x][d] = K[col * D + d];
                s_V[threadIdx.x][d] = V[col * D + d];
            }
        }
        else {
            for (int d = 0; d < D; d++) {
                s_K[threadIdx.x][d] = 0.0f;
                s_V[threadIdx.x][d] = 0.0f;
            }
        }
        __syncthreads();

        // Compute QK^T (Tile)
        float s_tile[TILE_SIZE];
        for (int j = 0; j < TILE_SIZE; j++) {
            float score = 0.0f;
            for (int d = 0; d < D; d++) {
                score += q_reg[d] * s_K[j][d];
            }
            s_tile[j] = score / sqrtf((float)D);
        }

        // Online Softmax Update
        float m_block = -1e20f;
        for (int j = 0; j < TILE_SIZE; j++) {
            if (t * TILE_SIZE + j < N)
                m_block = fmaxf(m_block, s_tile[j]);
        }

        float m_new = fmaxf(m, m_block);
        float correction = expf(m - m_new);

        for (int d = 0; d < D; d++) acc[d] *= correction;
        l *= correction;

        // Compute PV (Tile)
        for (int j = 0; j < TILE_SIZE; j++) {
            if (t * TILE_SIZE + j < N) {
                float p = expf(s_tile[j] - m_new);
                l += p;
                for (int d = 0; d < D; d++) {
                    acc[d] += p * s_V[j][d];
                }
            }
        }
        m = m_new;
        __syncthreads();
    }

    // Write Output
    for (int d = 0; d < D; d++) {
        O[row * D + d] = acc[d] / l;
    }
}

// =========================================================
// 3. CPU Implementation (검증용)
// =========================================================
void attention_cpu(float* Q, float* K, float* V, float* O, int N, int D) {
    std::vector<float> score(N);
    LOOP_I(N) {
        float max_val = -1e20f;
        LOOP_J(N) {
            float sum = 0.0f;
            LOOP_K(D) sum += Q[i * D + k] * K[j * D + k];
            score[j] = sum / sqrtf((float)D);
            if (score[j] > max_val) max_val = score[j];
        }
        float sum_exp = 0.0f;
        LOOP_J(N) {
            score[j] = expf(score[j] - max_val);
            sum_exp += score[j];
        }
        LOOP_K(D) {
            float out_val = 0.0f;
            LOOP_J(N) out_val += (score[j] / sum_exp) * V[j * D + k];
            O[i * D + k] = out_val;
        }
    }
}

int main() {
    const int N = N_SAMPLES;
    const int D = HEAD_DIM;
    size_t size_Matrix = N * D * sizeof(float);
    size_t size_Score = N * N * sizeof(float); // Naive용 Score 행렬 크기 (매우 큼)

    printf("Setting: N=%d, D=%d\n", N, D);
    printf("Score Matrix Size: %.2f MB\n", size_Score / (1024.0 * 1024.0));

    DS_timer timer(4);
    timer.setTimerName(0, "CPU_Full_Attention");
    timer.setTimerName(1, "GPU_Naive_Attention");
    timer.setTimerName(2, "GPU_Flash_Attention");
    timer.setTimerName(3, "Data_Transfer");

    float* h_Q = new float[N * D];
    float* h_K = new float[N * D];
    float* h_V = new float[N * D];
    float* h_O_cpu = new float[N * D];
    float* h_O_naive = new float[N * D];
    float* h_O_flash = new float[N * D];

    srand(2025);
    LOOP_I(N * D) {
        h_Q[i] = (float)rand() / RAND_MAX;
        h_K[i] = (float)rand() / RAND_MAX;
        h_V[i] = (float)rand() / RAND_MAX;
    }

    float* d_Q, * d_K, * d_V, * d_O, * d_Score;
    HANDLER_ERROR_ERR(cudaMalloc(&d_Q, size_Matrix));
    HANDLER_ERROR_ERR(cudaMalloc(&d_K, size_Matrix));
    HANDLER_ERROR_ERR(cudaMalloc(&d_V, size_Matrix));
    HANDLER_ERROR_ERR(cudaMalloc(&d_O, size_Matrix));
    HANDLER_ERROR_ERR(cudaMalloc(&d_Score, size_Score)); // Naive 전용 Global Memory

    // 1. CPU
    printf("1. Running CPU Attention...\n");
    timer.onTimer(0);
    attention_cpu(h_Q, h_K, h_V, h_O_cpu, N, D);
    timer.offTimer(0);

    // 데이터 복사
    timer.onTimer(3);
    HANDLER_ERROR_ERR(cudaMemcpy(d_Q, h_Q, size_Matrix, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_K, h_K, size_Matrix, cudaMemcpyHostToDevice));
    HANDLER_ERROR_ERR(cudaMemcpy(d_V, h_V, size_Matrix, cudaMemcpyHostToDevice));
    timer.offTimer(3);

    // 2. Naive GPU
    printf("2. Running GPU Naive Attention...\n");
    // Naive는 스레드당 1개 Row 처리 (블록 256 가정)
    dim3 block_naive(NAIVE_BLOCK_SIZE);
    dim3 grid_naive((N + NAIVE_BLOCK_SIZE) / NAIVE_BLOCK_SIZE);

    timer.onTimer(1);
    naive_attention_kernel << <grid_naive, block_naive >> > (d_Q, d_K, d_V, d_O, d_Score, N, D);
    cudaDeviceSynchronize();
    timer.offTimer(1);
    HANDLER_ERROR_ERR(cudaMemcpy(h_O_naive, d_O, size_Matrix, cudaMemcpyDeviceToHost));

    // 3. Flash GPU
    printf("3. Running GPU Flash Attention...\n");
    dim3 block_flash(TILE_SIZE);
    dim3 grid_flash((N + TILE_SIZE - 1) / TILE_SIZE);

    timer.onTimer(2);
    flash_attention_kernel << <grid_flash, block_flash >> > (d_Q, d_K, d_V, d_O, N, D);
    cudaDeviceSynchronize();
    timer.offTimer(2);
    HANDLER_ERROR_ERR(cudaMemcpy(h_O_flash, d_O, size_Matrix, cudaMemcpyDeviceToHost));

    // 검증
    bool naive_pass = true;
    bool flash_pass = true;
    float epsilon = EPSILON;

    LOOP_I(N * D) {
        if (fabs(h_O_cpu[i] - h_O_naive[i]) > epsilon) naive_pass = false;
        if (fabs(h_O_cpu[i] - h_O_flash[i]) > epsilon) flash_pass = false;
        if (!naive_pass && !flash_pass) break;
    }

    std::cout << "\n====================================" << std::endl;
    std::cout << "Verification Results:" << std::endl;
    std::cout << "Naive GPU: " << (naive_pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "Flash GPU: " << (flash_pass ? "PASS" : "FAIL") << std::endl;
    std::cout << "====================================" << std::endl;

    timer.printTimer();

    SAFE_DELETE_ARR(h_Q); SAFE_DELETE_ARR(h_K); SAFE_DELETE_ARR(h_V);
    SAFE_DELETE_ARR(h_O_cpu); SAFE_DELETE_ARR(h_O_naive); SAFE_DELETE_ARR(h_O_flash);
    cudaFree(d_Q); cudaFree(d_K); cudaFree(d_V); cudaFree(d_O); cudaFree(d_Score);

    return 0;
}