#include "cyhal.h"
#include "cybsp.h"
#include "cyhal_clock.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "cycfg_clocks.h"

#define SINE_FREQ        50
#define SAMPLING_FREQ    256
#define FFT_SIZES_COUNT  6
#define NUM_EXECUTIONS   10

#define RAM_FUNC __attribute__((section(".cy_ramfunc")))

// FFT sizes to benchmark
const int FFT_SIZES[FFT_SIZES_COUNT] = {32, 64, 128, 256, 512, 1024};

extern uint32_t __StackLimit;
extern uint32_t __StackTop;

// Prototypes
RAM_FUNC void init_dwt_all_counters(void);
RAM_FUNC uint32_t read_cycle_counter(void);
RAM_FUNC void fill_stack_pattern_to_sp(void);
RAM_FUNC uint32_t measure_stack_usage(void);
RAM_FUNC void generate_sine_wave_f32(float32_t* input, int N, float signal_freq, float sampling_freq);
void benchmark_arm_cfft_f32(arm_cfft_instance_f32* fft_instance, float32_t* input, uint32_t clkFastfreq,
                            uint32_t* cycle_count, uint32_t* instr_est, uint32_t* stack_used, float* exec_time_us);
void calculate_averages(uint32_t* cycle_counts, uint32_t* instr_counts, float* exec_time_us, uint32_t* stack_usages, int num_executions);

RAM_FUNC void init_dwt_all_counters(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; // Enable trace
    DWT->CTRL |= (1 << 0)  | (1 << 16) | (1 << 17) | (1 << 18) | (1 << 19) | (1 << 20);

    DWT->CYCCNT = 0;
    DWT->CPICNT = 0;
    DWT->EXCCNT = 0;
    DWT->SLEEPCNT = 0;
    DWT->LSUCNT = 0;
    DWT->FOLDCNT = 0;
}

RAM_FUNC uint32_t read_cycle_counter(void) {
    return DWT->CYCCNT;
}

RAM_FUNC void fill_stack_pattern_to_sp(void) {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) {
        *p++ = 0xAAAAAAAA;
    }
}

RAM_FUNC uint32_t measure_stack_usage(void) {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp && *p == 0xAAAAAAAA) {
        p++;
    }
    return ((uint32_t)sp - (uint32_t)p);
}

RAM_FUNC void generate_sine_wave_f32(float32_t* input, int N, float signal_freq, float sampling_freq) {
    for (int i = 0; i < N; i++) {
        float value = sinf(2 * M_PI * signal_freq * i / sampling_freq);
        input[2 * i] = value;  // Real part
        input[2 * i + 1] = 0;  // Imaginary part
    }
}

void benchmark_arm_cfft_f32(arm_cfft_instance_f32* fft_instance, float32_t* input, uint32_t clkFastfreq,
                            uint32_t* cycle_count, uint32_t* instr_est, uint32_t* stack_used, float* exec_time_us) {
    init_dwt_all_counters();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
    arm_cfft_f32(fft_instance, input, 0, 1);  // Forward FFT with bit reversal
    uint32_t end_cycles = read_cycle_counter();

    *cycle_count = end_cycles - start_cycles;

    *instr_est = *cycle_count
                 - DWT->CPICNT
                 - DWT->EXCCNT
                 - DWT->SLEEPCNT
                 - DWT->LSUCNT
                 + DWT->FOLDCNT;

    *stack_used = measure_stack_usage();

    float time_sec = (float)(*cycle_count) / clkFastfreq;
    *exec_time_us = time_sec * 1e6f;
}

void calculate_averages(uint32_t* cycle_counts, uint32_t* instr_counts, float* exec_time_us, uint32_t* stack_usages, int num_executions) {
    uint32_t total_cycles = 0, total_instr = 0, total_stack_used = 0;
    float total_exec_time_us = 0;

    for (int i = 0; i < num_executions; i++) {
        total_cycles += cycle_counts[i];
        total_instr += instr_counts[i];
        total_stack_used += stack_usages[i];
        total_exec_time_us += exec_time_us[i];
    }

    printf("\nAverages across %d executions:\n\r", num_executions);
    printf("Average Cycle Count = %lu\n\r", total_cycles / num_executions);
    printf("Average Estimated Instructions = %lu\n\r", total_instr / num_executions);
    printf("Average Execution Time = %.2f us\n\r", total_exec_time_us / num_executions);
    printf("Average Stack Used = %lu bytes\n\r", total_stack_used / num_executions);
}

RAM_FUNC int main(void) {
    __disable_irq();  // Disable all interrupts

    cy_rslt_t result = cybsp_init();
    if (result != CY_RSLT_SUCCESS) CY_ASSERT(0);

    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                                    CYBSP_DEBUG_UART_CTS, CYBSP_DEBUG_UART_RTS, CY_RETARGET_IO_BAUDRATE);
    if (result != CY_RSLT_SUCCESS) CY_ASSERT(0);

    printf("**************************************************\n\r");
    printf("FFT F32 Benchmark (Running from RAM)\n\r");

    uint32_t clkFastfreq = Cy_SysClk_ClkFastGetFrequency();
    printf("CPU Clock Frequency: %lu Hz\n", clkFastfreq);

    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        float32_t* input = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* original_input = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* magnitude = (float32_t*)malloc(N * sizeof(float32_t));
        float32_t* magnitude_reference = (float32_t*)malloc(N * sizeof(float32_t));
        if (!input || !original_input || !magnitude || !magnitude_reference) {
            printf("Memory allocation failed for FFT size N = %d\n\r", N);
            return -1;
        }

        // Generate sine wave
        generate_sine_wave_f32(original_input, N, SINE_FREQ, SAMPLING_FREQ);

        // Buffers for measurements
        uint32_t cycle_counts[NUM_EXECUTIONS];
        uint32_t instr_counts[NUM_EXECUTIONS];
        uint32_t stack_usages[NUM_EXECUTIONS];
        float exec_time_us_values[NUM_EXECUTIONS];

        // Initialize FFT instance
        arm_cfft_instance_f32 fft_instance;
        if (arm_cfft_init_f32(&fft_instance, N) != ARM_MATH_SUCCESS) {
            printf("FFT init failed for N = %d\n", N);
            free(input); free(original_input); free(magnitude); free(magnitude_reference);
            continue;
        }

        printf("\nFFT Size: %d\n", N);

        // Flag to check magnitude consistency
        int magnitudes_consistent = 1;

        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            memcpy(input, original_input, 2 * N * sizeof(float32_t));
            benchmark_arm_cfft_f32(&fft_instance, input, clkFastfreq,
                                   &cycle_counts[execution], &instr_counts[execution],
                                   &stack_usages[execution], &exec_time_us_values[execution]);

            // Compute magnitudes and check consistency
            arm_cmplx_mag_f32(input, magnitude, N);
            if (execution == 0) {
                memcpy(magnitude_reference, magnitude, N * sizeof(float32_t)); // Save reference magnitudes
//                printf("Frequency Bin, Magnitude (First Execution)\n\r");
//                for (int i = 0; i < N; i++) {
//                    float frequency_bin = (i * ((float32_t)SAMPLING_FREQ / (float32_t)N));
//                    printf("%.2f, %.4f\n\r", frequency_bin, magnitude[i]);
//                }
            } else {
                for (int i = 0; i < N; i++) {
                    if (fabsf(magnitude[i] - magnitude_reference[i]) > 1e-3) { // Adjust tolerance as necessary
                        magnitudes_consistent = 0;
                        break;
                    }
                }
            }
            printf("\n\r");
            printf("Execution %d: Cycle Count = %lu, Estimated Instructions = %lu, Time = %.2f us, Stack Used = %lu bytes\n\r",
                   execution + 1, cycle_counts[execution], instr_counts[execution],
                   exec_time_us_values[execution], stack_usages[execution]);
        }

        if (magnitudes_consistent) {
            printf("All magnitudes are consistent across executions for FFT size N = %d\n\r", N);
        } else {
            printf("Inconsistent magnitudes detected for FFT size N = %d\n\r", N);
        }

        calculate_averages(cycle_counts, instr_counts, exec_time_us_values, stack_usages, NUM_EXECUTIONS);

        // Free allocated memory
        free(input);
        free(original_input);
        free(magnitude);
        free(magnitude_reference);
    }

    printf("\nBenchmark completed.\n\r");
    return 0;
}
