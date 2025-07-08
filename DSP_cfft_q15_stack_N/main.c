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
#define Q15_SCALE        32768

#define RAM_FUNC __attribute__((section(".cy_ramfunc")))

const int FFT_SIZES[FFT_SIZES_COUNT] = {32, 64, 128, 256, 512, 1024};

extern uint32_t __StackLimit;
extern uint32_t __StackTop;

RAM_FUNC void init_dwt_all_counters(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;      // Enable trace
    DWT->CTRL |= (1 << 0);                               // CYCCNTENA
    DWT->CTRL |= (1 << 16);                              // CPICNT
    DWT->CTRL |= (1 << 17);                              // EXCCNT
    DWT->CTRL |= (1 << 18);                              // SLEEPCNT
    DWT->CTRL |= (1 << 19);                              // LSUCNT
    DWT->CTRL |= (1 << 20);                              // FOLDCNT

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
    while (p < sp) {
        if (*p != 0xAAAAAAAA) {
            break;
        }
        p++;
    }
    return ((uint32_t)sp - (uint32_t)p);
}

RAM_FUNC void generate_sine_wave_q15(q15_t* input, int N, float signal_freq, float sampling_freq) {
    for (int i = 0; i < N; i++) {
        float value = sinf(2 * M_PI * signal_freq * i / sampling_freq);
        input[2 * i] = (q15_t)(value * Q15_SCALE);  // Real
        input[2 * i + 1] = 0;                       // Imag
    }
}


RAM_FUNC int main(void) {
	__disable_irq();  // Disable all interrupts

    cy_rslt_t result;

    result = cybsp_init();
    if (result != CY_RSLT_SUCCESS) CY_ASSERT(0);


    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                                    CYBSP_DEBUG_UART_CTS, CYBSP_DEBUG_UART_RTS, CY_RETARGET_IO_BAUDRATE);
    if (result != CY_RSLT_SUCCESS) CY_ASSERT(0);

    printf("**************************************************\n\r");
    printf("FFT Q15 Benchmark (Running from RAM)\n\r");

    uint32_t clkFastfreq = Cy_SysClk_ClkFastGetFrequency();
    printf("CPU Clock Frequency: %lu Hz\n\r", clkFastfreq);

    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        q15_t* input = (q15_t*)malloc(2 * N * sizeof(q15_t));
        q15_t* output = (q15_t*)malloc(N * sizeof(q15_t));
        if (!input || !output) {
            printf("Memory allocation failed for FFT size = %d\n\r", N);
            free(input);
            free(output);
            continue;
        }

        generate_sine_wave_q15(input, N, SINE_FREQ, SAMPLING_FREQ);

        arm_cfft_instance_q15 fft_instance;
        if (arm_cfft_init_q15(&fft_instance, N) != ARM_MATH_SUCCESS) {
            printf("FFT init failed for N = %d\n\r", N);
            free(input);
            free(output);
            continue;
        }

        printf("\nFFT Size: %d\n\r", N);

        init_dwt_all_counters();
        fill_stack_pattern_to_sp();

        uint32_t start_cycles = read_cycle_counter();
        arm_cfft_q15(&fft_instance, input, 0, 1);  // forward FFT, bit reversal
        uint32_t end_cycles = read_cycle_counter();
        uint32_t cycle_count = end_cycles - start_cycles;

        uint32_t instr_est = cycle_count
                           - DWT->CPICNT
                           - DWT->EXCCNT
                           - DWT->SLEEPCNT
                           - DWT->LSUCNT
                           + DWT->FOLDCNT;

        uint32_t stack_used = measure_stack_usage();
        float time_sec = (float)cycle_count / clkFastfreq;
        float time_us = time_sec * 1e6f;

        arm_cmplx_mag_q15(input, output, N);

        printf("  Cycle Count       : %lu cycles\n\r", (unsigned long)cycle_count);
        printf("  Estimated Instr   : %lu\n\r", instr_est);
        printf("  Time (approx)     : %.3f us (%.6f s)\n\r", time_us, time_sec);
        printf("  Stack Used        : %lu bytes\n\r", (unsigned long)stack_used);

        free(input);
        free(output);
    }

    printf("\nBenchmark completed.\n\r");
    return 0;
}
