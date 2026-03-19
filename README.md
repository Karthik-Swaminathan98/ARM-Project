# ARM-Project

> Bare-metal benchmarking of **CMSIS-DSP and CMSIS-NN kernels** on ARM Cortex-M4 —
> cycle-accurate measurement using the **DWT (Data Watchpoint and Trace)** hardware
> cycle counter on real silicon (PSoC6 CY8CKIT-062-WIFI-BT).

Part of a Master's thesis at **TU Chemnitz** in collaboration with
**Infineon Technologies**, Dresden (2025).

Companion RISC-V repo: [RISCV-Project](https://github.com/Karthik-Swaminathan98/RISCV-Project)  
Full cross-architecture results: [arm-riscv-benchmark-results](https://github.com/Karthik-Swaminathan98/arm-riscv-benchmark-results)

![Language](https://img.shields.io/badge/language-C-blue)
![Platform](https://img.shields.io/badge/platform-ARM%20Cortex--M4-informational)
![Libraries](https://img.shields.io/badge/libraries-CMSIS--DSP%20%7C%20CMSIS--NN-orange)
![Board](https://img.shields.io/badge/board-PSoC6%20CY8CKIT--062-lightgrey)
![Thesis](https://img.shields.io/badge/thesis-TU%20Chemnitz%20%2F%20Infineon%202025-green)

---

## Hardware platform

| Feature | CY8CKIT-062-WIFI-BT (PSoC6) |
|---|---|
| Core | ARM Cortex-M4 (Armv7E-M) |
| Pipeline | 3-stage in-order |
| Clock | 150 MHz max · **25 MHz used for benchmarking** |
| SRAM | 288 KB |
| Flash | 1 MB |
| Key extensions | FPU · DSP · SIMD |
| Toolchain | GNU ARM Embedded 13.3 |
| IDE | ModusToolbox 3.4 |
| Programmer | KitProg3 (SWD) |

All benchmarks execute **entirely from RAM** — flash wait-states eliminated via
`.cy_ramfunc` section attribute. Interrupts disabled during measurement.
Each kernel built and flashed as a standalone ModusToolbox project.

---

## What is measured

Five metrics are captured per kernel:

| Metric | Method |
|---|---|
| Cycle count | DWT->CYCCNT hardware register |
| Instruction count | DWT auxiliary counters (CPICNT, EXCCNT, SLEEPCNT, LSUCNT, FOLDCNT) |
| Execution time | Derived from cycle count ÷ core frequency |
| Stack usage | Stack-paint technique (0xAAAAAAAA pattern) |
| Code size | .map file + objdump call-graph analysis |

---

## Measurement implementation

### Cycle count — DWT hardware counter

```c
/* Enable DWT cycle counter */
CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
DWT->CYCCNT = 0;
DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;

uint32_t start = DWT->CYCCNT;
benchmark_function();
uint32_t cycle_count = DWT->CYCCNT - start;
```

### Instruction count — DWT auxiliary counters

ARM Cortex-M4 does not have a direct instruction counter.
Instruction count is estimated using the DWT auxiliary registers:

```c
/* Reset all DWT counters */
DWT->CPICNT   = 0;
DWT->EXCCNT   = 0;
DWT->SLEEPCNT = 0;
DWT->LSUCNT   = 0;
DWT->FOLDCNT  = 0;

/* Estimation formula (ARM-recommended) */
uint32_t instr_count = cycle_count
    - DWT->CPICNT
    - DWT->EXCCNT
    - DWT->SLEEPCNT
    - DWT->LSUCNT
    + DWT->FOLDCNT;
```

### Execution time

```c
/* PSoC6 SDK clock query */
clkFastfreq = Cy_SysClk_ClkFastGetFrequency();
float time_us = ((float)cycle_count / clkFastfreq) * 1e6f;
```

### Stack usage — stack-paint technique

```c
/* RAM execution via section attribute */
#define RAM_FUNC __attribute__((section(".cy_ramfunc")))

void fill_stack_pattern_to_sp(void) {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) { *p++ = 0xAAAAAAAA; }
}

uint32_t measure_stack_usage(void) {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp && *p == 0xAAAAAAAA) { p++; }
    return ((uint32_t)sp - (uint32_t)p);
}
```

### Complete benchmark loop

```c
fill_stack_pattern_to_sp();
enable_cycle_counter();

uint32_t start_cycles = read_cycle_counter();
arm_convolve_wrapper_s8(&ctx, &conv_params, ...);
uint32_t end_cycles = read_cycle_counter();

uint32_t cycle_count = end_cycles - start_cycles;
uint32_t instr_count = cycle_count
    - DWT->CPICNT - DWT->EXCCNT
    - DWT->SLEEPCNT - DWT->LSUCNT
    + DWT->FOLDCNT;

uint32_t stack_used = measure_stack_usage();
float time_us = ((float)cycle_count / clkFastfreq) * 1e6f;
```

---

## Kernels and models benchmarked

### DSP — CMSIS-DSP

| Project | Kernel | Data types |
|---|---|---|
| `DSP_FFT_benchmark` | Complex FFT (CFFT) | F32, Q15 |
| `DSP_FIR_benchmark` | FIR filter | F32, Q15 |
| `DSP_Mag_benchmark` | Complex magnitude | F32, Q15 |
| `DSP_Math_benchmark` | Fast math (sqrt, sin, cos, atan2) | F32, Q15 |

### Neural Network — CMSIS-NN

| Project | Operator | Data types |
|---|---|---|
| `NN_Activation_benchmark` | ReLU6, activation S16 | S8, S16 |
| `NN_convolution_benchmark` | Conv wrapper, depthwise conv | S8, S16 |
| `NN_fully_connected_benchmark` | Fully connected, per-channel | S8, S16 |
| `NN_Pooling_benchmark` | Average pooling | S8, S16 |
| `NN_Softmax_Benchmark` | Softmax | S8, S16 |
| `NN_LSTM_benchmark` | LSTM | S8, S16 |
| `NN_Transpose_benchmark` | Transpose convolution | S8, S16 |

### Model inference

| Project | Model | Task |
|---|---|---|
| `CIFAR10` | Quantised CNN (INT8) | CIFAR-10 image classification |
| `NN_KWS_DSCNN_SMALL` | DS-CNN Small (INT8) | Keyword spotting — 10 classes |
| `NN_KWS_DSCNN_MEDIUM` | DS-CNN Medium (INT8) | Keyword spotting — 10 classes |

---

## Build configuration

All kernels built in Release mode:

```
-O3
-ffunction-sections -fdata-sections
-flto
-Wl,-gc-sections
```

RAM execution enabled via linker + section attribute:

```c
#define RAM_FUNC __attribute__((section(".cy_ramfunc")))
RAM_FUNC int main(void) { ... }
```

### Libraries

| Library | Source |
|---|---|
| CMSIS-DSP v1.10 | [github.com/ARM-software/CMSIS_6](https://github.com/ARM-software/CMSIS_6) |
| CMSIS-NN | [github.com/ARM-software/CMSIS-NN](https://github.com/ARM-software/CMSIS-NN) |

NN test vectors sourced from the official CMSIS-NN test suite — output validated
against golden reference vectors for correctness.

---

## Key results (vs CMSIS baseline = 1.0)

ARM CMSIS is the baseline — this table shows where RISC-V outperforms or underperforms.

| Function | RISC-V speedup | Note |
|---|---|---|
| FFT F32 (Andes) | 1.47× faster | |
| Magnitude Q15 (Andes) | 3.32× faster | Largest RISC-V win |
| FIR F32 (CMSIS wins) | 0.74× | ARM 8-tap unrolling advantage |
| Convolution S8 (NMSIS) | 1.31× faster | |
| Activation S8 (CMSIS wins) | 0.78× | ARM conditional execution advantage |

Full results across all 5 metrics: [arm-riscv-benchmark-results](https://github.com/Karthik-Swaminathan98/arm-riscv-benchmark-results)

---

## How to build and run

**Requirements:**
- ModusToolbox 3.4
- GNU ARM Embedded Toolchain 13.3
- PSoC6 CY8CKIT-062-WIFI-BT evaluation board
- KitProg3 programmer (USB)

**Steps:**
1. Open ModusToolbox 3.4
2. File → Import → Existing Projects into Workspace
3. Select any project folder (e.g. `NN_convolution_benchmark`)
4. Select Release configuration → Build
5. Flash via KitProg3 USB
6. Open serial terminal at 115200 baud

**Sample output:**
```
[BENCH] arm_convolve_wrapper_s8
  Cycles      : 63284
  Instructions: 58941
  Stack used  : 344 bytes
  Exec time   : 2531.4 us
```

---

## Related repos

| Repo | Description |
|---|---|
| [RISCV-Project](https://github.com/Karthik-Swaminathan98/RISCV-Project) | RISC-V counterpart — NMSIS-DSP/NN benchmarks on Telink B91 |
| [arm-riscv-benchmark-results](https://github.com/Karthik-Swaminathan98/arm-riscv-benchmark-results) | Full cross-architecture results — DSP, NN, and model inference |
| [mcu-function-size-analyser](https://github.com/Karthik-Swaminathan98/mcu-function-size-analyser) | Python tool — dependency-aware function code size analyser |

---

## Acknowledgements

Master's thesis at **Technische Universität Chemnitz**
(Chair of Computer Architectures and Systems)
in collaboration with **Infineon Technologies**, Dresden.

Supervised by Prof. Dr. Alejandro Masrur · Mr. Daniel Markert ·
Dr. Elias Trommer · Mr. Jerome Almon Swamidasan

---

## Author

**Karthik Swaminathan** — Embedded Firmware Engineer  
M.Sc. Embedded Systems · TU Chemnitz  
[LinkedIn](https://linkedin.com/in/karthik-swaminathan98) · karthik94870@gmail.com
