# CUDA Attention 최적화 프로젝트 (Naive vs. Flash Attention)

이 프로젝트는 Transformer 모델의 핵심 연산인 **Attention 메커니즘**을 세 가지 방식(CPU, GPU Naive, GPU Flash)으로 구현하고 그 성능과 정확도를 비교하는 것을 목적으로 합니다.

## 1. 프로젝트 개요

* 
**CPU Attention**: 3중 루프를 사용한 기본적인 행렬 연산 구현.


* 
**GPU Naive Attention**: Global Memory에 중간 결과(Score 행렬)를 직접 읽고 쓰는 방식.


* 
**GPU Flash Attention**: **Tiling**과 **Online Softmax** 기법을 사용하여 SRAM(Shared Memory) 활용을 극대화하고 메모리 I/O를 최적화한 방식.



## 2. 사전 준비 사항

프로젝트를 구동하기 위해서는 다음 환경이 필요합니다:

* **OS**: Windows (Visual Studio) 또는 Linux (GCC).
* **GPU**: NVIDIA GPU (Compute Capability 지원 모델).
* **Toolkit**: **CUDA Toolkit** 설치 필요.
* **Compiler**: `nvcc` (NVIDIA CUDA Compiler).

## 3. 파일 구성

* `kernel.cu`: 메인 소스 코드 (커널 구현 및 메인 함수 포함).
* `DS_timer.h / .cpp`: 성능 측정을 위한 타이머 클래스.
* `DS_definitions.h`: 프로젝트 공통 매크로 및 유틸리티 정의.
* `Error.h`: CUDA 에러 핸들링 유틸리티.

## 4. 빌드 및 실행 방법

### Linux / Terminal 환경

`nvcc`를 사용하여 직접 컴파일하는 방법입니다.

```bash
# 1. 소스 코드 컴파일
nvcc -o attention_test kernel.cu DS_timer.cpp -I.

# 2. 프로그램 실행
./attention_test

```

### Windows (Visual Studio) 환경

1. Visual Studio에서 새로운 **CUDA Runtime Project**를 생성합니다.
2. 프로젝트 폴더에 모든 소스 파일(`.cu`, `.cpp`, `.h`)을 복사합니다.
3. `DS_timer.cpp`를 프로젝트의 '소스 파일' 항목에 추가합니다.
4. **Solution Configuration**을 `Release`, **Platform**을 `x64`로 설정한 후 빌드(Ctrl+Shift+B) 및 실행(F5)합니다.

## 5. 주요 파라미터 설정

실험 환경을 변경하려면 `kernel.cu` 상단의 매크로 값을 수정하십시오:

| 매크로 명칭 | 설명 | 기본값 |
| --- | --- | --- |
| `N_SAMPLES` | 시퀀스 길이 (Input Sequence Length) | 8192 |
| `HEAD_DIM` | 헤드 차원 (Dimension of Q, K, V) | 32 |
| `TILE_SIZE` | Flash Attention 연산 시 타일 크기 | 32 |
| `NAIVE_BLOCK_SIZE` | Naive 커널의 블록 크기 | 256 |

> 
> **주의**: `HEAD_DIM`이나 `TILE_SIZE`를 너무 크게 설정할 경우, GPU 레지스터 용량 초과로 인해 성능이 저하되거나 검증에 실패할 수 있습니다.
> 
> 

## 6. 출력 결과 확인

프로그램을 실행하면 다음과 같은 결과가 출력됩니다:

1. **Setting 정보**: 설정된 과  값, Score 행렬의 메모리 크기.
2. **Verification**: CPU 결과와 GPU 결과 간의 오차 검증 (PASS/FAIL).
3. **Timer Report**: 각 방식별 실행 시간(ms) 측정 결과.