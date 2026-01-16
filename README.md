# GPU 가속을 통한 Attention 최적화: Naive vs. Flash Attention 비교

본 프로젝트는 Transformer 모델의 핵심 연산인 Attention 메커니즘을 CUDA를 통해 구현하고, 메모리 접근 방식의 차이가 성능에 미치는 영향을 분석합니다. 특히 기존 Naive 방식의 메모리 병목 현상을 해결한 Flash Attention의 이론적 원리를 직접 구현하여 성능을 비교하는 데 목적이 있습니다.

---

## 1. 프로젝트 개요

### Attention 연산의 병목 문제

* **Naive Attention**: Query(Q)와 Key(K)의 행렬곱 결과인  크기의 Score 행렬을 Global Memory(VRAM)에 쓰고 다시 읽는 과정을 반복합니다.


* **복잡도**: 시퀀스 길이()가 늘어날수록 $O(N^2)$의 시간 및 공간 복잡도가 발생하여 메모리 I/O 병목이 심화됩니다.



### Flash Attention의 해결책

* **Tiling**: 데이터를 타일 단위로 쪼개어 속도가 빠른 Shared Memory(SRAM) 내에서 연산을 수행합니다.


* **Online Softmax**: 전체 데이터를 다 보지 않고도 Softmax를 계산할 수 있도록 타일을 읽을 때마다 중간 계산 값을 보정하며 메모리 접근을  수준으로 최적화합니다.



---

## 2. 주요 구현 커널 설명

### 1) CPU Attention (Baseline)

* **특징**: 검증을 위한 표준 구현체로, 3중 Loop를 사용한 비효율적 연산 방식입니다.


* **역할**: GPU 연산 결과의 정확도(Verification)를 판단하는 기준이 됩니다.



### 2) GPU Naive Attention

* **특징**: 중간 단계의 Score 행렬을 Global Memory에 저장하고 읽어오는 방식입니다.


* **단점**: 단계별(MatMul -> Softmax -> MatMul)로 빈번한 Global Memory 쓰기/읽기가 발생하여 성능이 저하됩니다.



### 3) GPU Flash Attention

* **SRAM 활용**: 한 번 로드된 Key(K), Value(V) 타일을 블록 내 스레드가 공유하여 읽기 횟수를 줄입니다.


* **Register Caching**: Query(Q) 데이터를 레지스터에 캐싱하여 사용합니다.


* **Memory Efficiency**:  크기의 Score 행렬을 별도로 저장하지 않고 부분합 변수만 유지합니다.



---

## 3. 구동 방법 (Environment & Build)

### 사전 준비 사항

* **OS**: Windows (Visual Studio 2022 권장) 또는 Linux.
* **GPU**: NVIDIA GPU (CUDA 지원 모델).
* **Toolkit**: CUDA Toolkit 설치 완료 상태.

### 빌드 및 실행 (Command Line)

`nvcc` 컴파일러를 사용하여 프로젝트를 빌드합니다.

```bash
# 1. 소스 코드 컴파일
# kernel.cu와 DS_timer.cpp를 함께 컴파일합니다.
nvcc -o attention_benchmark kernel.cu DS_timer.cpp -I.

# 2. 프로그램 실행
./attention_benchmark

```

### 파라미터 수정 (`kernel.cu`)

실험 환경을 변경하려면 코드 상단의 매크로 상수를 수정하십시오.

* `N_SAMPLES`: 시퀀스 길이 (기본값: 8192).
* `HEAD_DIM`: 헤드 차원 (기본값: 32).
* `TILE_SIZE`: Flash Attention 타일 크기 (기본값: 32).

---

## 4. 실험 결과 분석 (보고서 요약)

시퀀스 길이()에 따른 성능 

| 시퀀스 길이 () | 성능 분석 | 결과 |
| --- | --- | --- |
| **512** | 데이터가 작을 때 Tiling 효과가 극대화됨 | Flash 1.62배 우세 |
| **1024** | 여전히 Flash Attention이 효율적임 | Flash 1.16배 우세 |
| **8192** | 특정 조건에서 성능 저하 발생 가능 (하드웨어 제약) | Naive 우세 사례 발생 |

### 주요 분석 포인트

* **하드웨어 제약**: 구현된 Flash Attention 커널이 스레드당 많은 레지스터(`q_reg[128]`, `acc[128]`)를 사용할 경우, 레지스터 용량 초과로 데이터가 VRAM으로 흘러넘쳐(Spilling) 성능이 저하될 수 있습니다.


* **정밀도 문제**: 타일 크기와 헤드 차원이 너무 클 경우 부동소수점 정밀도 오차로 인해 검증(Verification)이 실패할 수 있습니다.


* **최적화 결론**: 보유한 GPU 사양에 맞춰 타일 크기와 레지스터 사용량을 미세하게 조정하는 하드웨어 최적화(I/O Awareness)가 필수적입니다.
