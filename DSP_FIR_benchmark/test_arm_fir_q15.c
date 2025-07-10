#include "main.h"

RAM_FUNC void benchmark_fir_q15(void) {
    printf("=== FIR Q15 Benchmark (sine input, various N) ===\n\r");
    for (int idx = 0; idx < FIR_SIZES_COUNT; idx++) {
        int N = FIR_SIZES[idx];

        q15_t *input = (q15_t*)malloc(N * sizeof(q15_t));
        q15_t *output = (q15_t*)malloc(N * sizeof(q15_t));
        q15_t *firStateQ15 = (q15_t*)calloc(NUM_TAPS_q15 + N - 1, sizeof(q15_t));
        if (!input || !output || !firStateQ15) {
            printf("Memory allocation failed for N = %d\n\r", N);
            if (input) free(input);
            if (output) free(output);
            if (firStateQ15) free(firStateQ15);
            continue;
        }
        // Generate N-length sine wave input (Q15)
        for (int i = 0; i < N; i++) {
            float val = sinf(2 * M_PI * SINE_FREQ * i / SAMPLING_FREQ);
            input[i] = (q15_t)(val * Q15_SCALE);
//            if (q15val > 32767) q15val = 32767;
//            if (q15val < -32768) q15val = -32768;
//            input[i] = (q15_t)q15val;
        }

        arm_fir_instance_q15 S;
        arm_fir_init_q15(&S, NUM_TAPS_q15, (q15_t*)firCoeffsQ15, firStateQ15, N);

        enable_cycle_counter();
        fill_stack_pattern_to_sp();
        uint32_t start_cycles = read_cycle_counter();

        // Process whole input as one block (blockSize = N)
        arm_fir_q15(&S, input, output, N);

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

        printf("\nFIR Q15 N = %d (Sine input)\n\r", N);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);

        free(input);
        free(output);
        free(firStateQ15);
    }
    printf("\nBenchmark completed for ARM FIR Q15.\n\r");
}
