#include "main.h"

RAM_FUNC void benchmark_fir_f32(void) {
    printf("=== FIR F32 Benchmark (sine input, various N) ===\n\r");
    for (int idx = 0; idx < FIR_SIZES_COUNT; idx++) {
        int N = FIR_SIZES[idx];

        float32_t *input = (float32_t*)malloc(N * sizeof(float32_t));
        float32_t *output = (float32_t*)malloc(N * sizeof(float32_t));
        float32_t *firStateF32 = (float32_t*)calloc(NUM_TAPS + N - 1, sizeof(float32_t));
        if (!input || !output || !firStateF32) {
            printf("Memory allocation failed for N = %d\n\r", N);
            if (input) free(input);
            if (output) free(output);
            if (firStateF32) free(firStateF32);
            continue;
        }

        // Generate N-length sine wave input
        for (int i = 0; i < N; i++)
            input[i] = sinf(2 * M_PI * SINE_FREQ * i / SAMPLING_FREQ);

        arm_fir_instance_f32 S;
        arm_fir_init_f32(&S, NUM_TAPS, (float32_t*)firCoeffs32, firStateF32, N);

        enable_cycle_counter();
        fill_stack_pattern_to_sp();
        uint32_t start_cycles = read_cycle_counter();

        // Process whole input as one block (blockSize = N)
        arm_fir_f32(&S, input, output, N);

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

        printf("\nFIR N = %d\n\r", N);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);

        free(input);
        free(output);
        free(firStateF32);
    }
    printf("\nBenchmark completed for ARM FIR F32.\n\r");
}
