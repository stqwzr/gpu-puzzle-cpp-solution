# 🧩 GPU Puzzles — C++ CUDA Solutions

C++ CUDA solutions for [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) by Sasha Rush.

The original project teaches GPU programming through puzzles using Python and Numba. This repo re-implements all solutions in **native CUDA C++** — closer to the metal, closer to real-world GPU development.

## Requirements

- NVIDIA GPU (tested on RTX 4090)
- CUDA Toolkit 13.1+
- CMake 3.18+
- GCC 13 (as host compiler)

## Build

```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug ..
make
```

## Puzzles

| #  | Puzzle | Concept | Status |
|----|--------|---------|--------|
| 1  | Map | One thread per element, `threadIdx.x` | ✅ |
| 2  | Zip | Two input arrays, element-wise operation | ✅ |
| 3  | Guards | Boundary checking with `if` | ✅ |
| 4  | Map 2D | 2D indexing, `threadIdx.x/y`, flat memory | ✅ |
| 5  | Broadcast | Vector broadcast with 2D output | ✅ |
| 6  | Blocks | Multiple blocks, global index formula | ✅ |
| 7  | Blocks 2D | 2D blocks + 2D threads | ✅ |
| 8  | Shared Memory | `__shared__`, `__syncthreads()` | ✅ |
| 9  | Pooling | Shared memory reductions |  ✅ |
| 10 | Dot Product | Block-level reduction | 🔧 |
| 11 | 1D Convolution | Sliding window with shared memory | ⬚ |
| 12 | 2D Convolution | 2D tiling and halo regions | ⬚ |
| 13 | Matrix Multiply | Tiled matmul with shared memory | ⬚ |

## Key CUDA Concepts Covered

**Execution Model** — threads, blocks, grids, `dim3` launch configuration

**Memory Hierarchy** — global memory → shared memory → registers

**Indexing** — `threadIdx`, `blockIdx`, `blockDim`, `gridDim`, flat 2D indexing (`i * size + j`)

**Synchronization** — `__syncthreads()` barriers, avoiding race conditions

**Error Handling** — `cudaGetLastError()`, `cudaDeviceSynchronize()`


## Acknowledgments

Based on [GPU-Puzzles](https://github.com/srush/GPU-Puzzles) by [Sasha Rush](https://github.com/srush). Original puzzles use Python/Numba — this project adapts them to CUDA C++ for a lower-level learning experience.

## License

MIT
