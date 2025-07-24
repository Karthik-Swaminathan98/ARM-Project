#include "main.h"

RAM_FUNC void benchmark_sin_cos(void) {
    printf("=== SIN/COS Benchmark ===\n\r");
    for (int i = 0; i < FIR_SIZES_COUNT; i++) {
        int N = FIR_SIZES[i];

        float32_t *ang_f32 = malloc(N*sizeof(float32_t));
        float32_t *s_f32   = malloc(N*sizeof(float32_t));
        float32_t *c_f32   = malloc(N*sizeof(float32_t));
        q15_t     *ang_q15 = malloc(N*sizeof(q15_t));
        q15_t     *s_q15   = malloc(N*sizeof(q15_t));
        q15_t     *c_q15   = malloc(N*sizeof(q15_t));
        if (!ang_f32||!s_f32||!c_f32||!ang_q15||!s_q15||!c_q15) {
            printf("Mem alloc failed for SIN/COS N=%d\n\r", N);
            free(ang_f32); free(s_f32); free(c_f32);
            free(ang_q15); free(s_q15); free(c_q15);
            continue;
        }

        // Inputs: angles [0..2π)
        for (int j = 0; j < N; j++) {
            float32_t a    = 2*M_PI * j / N;
            ang_f32[j]     = a;
            ang_q15[j]     = (q15_t)((a/(2*M_PI))*0x7FFF);
        }

        // ---- f32 sin ----
        enable_cycle_counter();
        fill_stack_pattern_to_sp();
        uint32_t start_cycles = read_cycle_counter();

        for (int j = 0; j < N; j++) {
            s_f32[j] = arm_sin_f32(ang_f32[j]);
        }

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

        printf("\nSIN f32 N = %d\n\r", N);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);

        // ---- f32 cos ----
        enable_cycle_counter();
        fill_stack_pattern_to_sp();
        start_cycles = read_cycle_counter();

        for (int j = 0; j < N; j++) {
            c_f32[j] = arm_cos_f32(ang_f32[j]);
        }

        end_cycles = read_cycle_counter();
        cycle_count = end_cycles - start_cycles;
        instr_est = cycle_count
                           - DWT->CPICNT
                           - DWT->EXCCNT
                           - DWT->SLEEPCNT
                           - DWT->LSUCNT
                           + DWT->FOLDCNT;
        stack_used = measure_stack_usage();
        time_sec = (float)cycle_count / clkFastfreq;
        time_us = time_sec * 1e6f;

        printf("\nCOS f32  N = %d\n\r", N);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);

        // ---- Q15 sin ----
        enable_cycle_counter();
        fill_stack_pattern_to_sp();
        start_cycles = read_cycle_counter();

        for (int j = 0; j < N; j++) {
            s_q15[j] = arm_sin_q15(ang_q15[j]);
        }

        end_cycles = read_cycle_counter();
        cycle_count = end_cycles - start_cycles;
        instr_est = cycle_count
                           - DWT->CPICNT
                           - DWT->EXCCNT
                           - DWT->SLEEPCNT
                           - DWT->LSUCNT
                           + DWT->FOLDCNT;
        stack_used = measure_stack_usage();
        time_sec = (float)cycle_count / clkFastfreq;
        time_us = time_sec * 1e6f;

        printf("\nSIN Q15 N = %d\n\r", N);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);

        // ---- Q15 cos ----
        enable_cycle_counter();
        fill_stack_pattern_to_sp();
        start_cycles = read_cycle_counter();

        for (int j = 0; j < N; j++) {
            c_q15[j] = arm_cos_q15(ang_q15[j]);
        }

        end_cycles = read_cycle_counter();
        cycle_count = end_cycles - start_cycles;
        instr_est = cycle_count
                           - DWT->CPICNT
                           - DWT->EXCCNT
                           - DWT->SLEEPCNT
                           - DWT->LSUCNT
                           + DWT->FOLDCNT;
        stack_used = measure_stack_usage();
        time_sec = (float)cycle_count / clkFastfreq;
        time_us = time_sec * 1e6f;

        printf("\nCOS Q15 N = %d\n\r", N);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);

        free(ang_f32); free(s_f32); free(c_f32);
        free(ang_q15); free(s_q15); free(c_q15);
    }
    printf("=== SIN/COS Benchmark Done ===\n\r");
}
