# RedeNeural

Neural network built from scratch in C++/CUDA — no ML libraries, no abstractions. The goal is to understand each layer, from the math to the hardware instruction.

## What it is

A personal research project structured in incremental optimization phases. Each phase deliberately exposes a deeper level of the hardware.

**Current phase:** all matrix operations (add, subtract, multiply) run entirely on the GPU via custom CUDA kernels, with manual host/device memory management and result visualization via Raylib.

**Next phases planned:**
- Shared memory to optimize matrix multiplication and reduce VRAM usage
- Tensor Core utilization (RTX 4050) for high-performance linear algebra

## Architecture

```
Include/
├── Matrix/
│   ├── Matrix.h             # n×m matrix with Host/Device modes
│   ├── ProcessingType.h     # CPU vs GPU dispatch enum
│   └── TensorComputeCore.h  # CUDA kernel interface
└── Artificial_Neural_Network/
    └── Layer.h              # Layer abstraction (weights, bias, activation)

Src/
├── Matrix/
│   ├── Matrix.cpp
│   └── TensorComputeCore.cu # GPU matrix multiply kernel
└── Artificial_Neural_Network/
    └── Layer.cpp
```

## Stack

- **C++20 / CUDA 17** — core implementation
- **Raylib** — matrix visualization
- **OpenMP** — CPU parallelism fallback
- Target: NVIDIA RTX 4050 (compute capability 8.9)

## Why CUDA

Academic recommendation and de facto standard for scientific computing on NVIDIA hardware. The absence of any ML library is intentional — the point is to understand what those libraries abstract away.
