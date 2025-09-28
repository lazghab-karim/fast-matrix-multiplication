# Fast Matrix Multiplication in C

This repository contains three different implementations of matrix multiplication in C, showcasing progressively faster approaches:

1. **Naive (sequential)** → Straightforward triple-nested loop implementation.  
2. **Parallel (OpenMP)** → Multi-threaded version using OpenMP.  
3. **SIMD (AVX2)** → Vectorized version using AVX2 + FMA intrinsics.  

The aim is to compare performance across these methods.

---

## 📂 Project Structure
```bash
.
├── naive.c # Baseline sequential matrix multiplication
├── parallel.c # OpenMP-based parallel matrix multiplication
├── SIMD.c # SIMD AVX2/FMA matrix multiplication
├── README.md # Documentation
```

## 🚀 Implementations

### 1. Naive (Sequential)
- Simple `O(n³)` approach with three nested loops.
- Acts as a performance baseline.

### 2. Parallel (OpenMP)
- Uses `#pragma omp parallel for` to parallelize loops.
- Exploits multiple CPU cores for faster execution.

### 3. SIMD (AVX2)
- Uses Intel AVX2 intrinsics (`__m256`, `_mm256_*`).
- Processes 8 floating-point values in a single instruction.
- Uses FMA (Fused Multiply-Add) for higher throughput.
- Requires an AVX2-capable CPU.

---

## ⚙️ Build Instructions

Compile each version separately using **gcc**:

```bash
# Naive version
gcc naive.c -o naive

# Parallel (OpenMP)
gcc -fopenmp parallel.c -o parallel

# SIMD (AVX2 + FMA)
gcc -mfma -mavx2 SIMD.c -o SIMD

```
## ▶️ Usage
Each executable multiplies two randomly generated N × N matrices.

```bash
./naive
./parallel
./SIMD.exe   # Windows (MinGW/MSYS2)
```
## 📊 Performance Overview

https://github.com/lazghab-karim/fast-matrix-multiplication/tree/main/KPI

Naive → Slowest, simple reference.

Parallel → Exploits multi-core CPUs; large speedup on systems with many threads.

SIMD → Fastest single-threaded execution; leverages vectorization.

Combining OpenMP + SIMD would yield even better results.

## 🔧 Requirements

- GCC or Clang with support for:

  - OpenMP (-fopenmp)

  - AVX2 + FMA (-mfma -mavx2)

- CPU with AVX2 support (Intel Haswell+ or AMD Excavator+).

- Works on Linux, macOS, or Windows (MinGW/MSYS2).
