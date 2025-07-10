#include "main.h"

RAM_FUNC void benchmark_ifft_q15() {
    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        q15_t* input           = (q15_t*)malloc(2 * N * sizeof(q15_t));
        q15_t* original_input  = (q15_t*)malloc(2 * N * sizeof(q15_t));
        q15_t* freq_domain     = (q15_t*)malloc(2 * N * sizeof(q15_t));
        if (!input || !original_input || !freq_domain) {
            printf("Memory allocation failed for FFT size N = %d\n\r", N);
            continue;
        }

        generate_sine_wave_q15(original_input, N, SINE_FREQ, SAMPLING_FREQ);

        // Step 1: Forward FFT to create freq_domain input
        memcpy(input, original_input, 2 * N * sizeof(q15_t));
        arm_cfft_instance_q15 fft_instance_fwd;
        arm_cfft_init_q15(&fft_instance_fwd, N);
        arm_cfft_q15(&fft_instance_fwd, input, 0, 1);  // Forward FFT
        memcpy(freq_domain, input, 2 * N * sizeof(q15_t));

        // Step 2: Prepare IFFT instance
        arm_cfft_instance_q15 fft_instance_inv;
        if (arm_cfft_init_q15(&fft_instance_inv, N) != ARM_MATH_SUCCESS) {
            printf("IFFT init failed for N = %d\n", N);
            free(input); free(original_input); free(freq_domain);
            continue;
        }

        uint32_t cycle_counts[NUM_EXECUTIONS];
        uint32_t instr_counts[NUM_EXECUTIONS];
        uint32_t stack_usages[NUM_EXECUTIONS];
        float exec_time_us_values[NUM_EXECUTIONS];

        int signals_consistent = 1;
        printf("\nIFFT Q15 Size: %d\n", N);

        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            memcpy(input, freq_domain, 2 * N * sizeof(q15_t));

            fill_stack_pattern_to_sp();
            enable_cycle_counter();
            uint32_t start_cycles = read_cycle_counter();

            arm_cfft_q15(&fft_instance_inv, input, 1, 1);  // Inverse FFT

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

            cycle_counts[execution] = cycle_count;
            instr_counts[execution] = instr_est;
            stack_usages[execution] = stack_used;
            exec_time_us_values[execution] = time_us;

            // Print first 8 outputs and references, scaled for comparison
            if (execution == 0) {
                printf("Index\tIFFT Output (scaled)\tReference\tOutput (float)\tReference (float)\n\r");
                for (int i = 0; i < (N < 8 ? N : 8); i++) {
                    int32_t output_scaled = (int32_t)input[2 * i] * N;
                    float output_f = (float)output_scaled / Q15_SCALE;
                    float ref_f = (float)original_input[2 * i] / Q15_SCALE;
                    printf("%2d\t%10ld\t\t%6d\t\t%.6f\t\t%.6f\n\r", i, output_scaled, original_input[2 * i], output_f, ref_f);
                }
            }

            // Consistency check: scale IFFT output by N
            for (int i = 0; i < N; i++) {
                int32_t output_scaled = (int32_t)input[2 * i] * N;
                int32_t diff = output_scaled - original_input[2 * i];
                if (diff > 4 || diff < -4) { // allow for rounding
                    signals_consistent = 0;
                    break;
                }
            }
        }

        if (signals_consistent) {
            printf("All IFFT Q15 outputs are consistent across executions for FFT size N = %d\n\r", N);
        } else {
            printf("Inconsistent IFFT Q15 outputs detected for FFT size N = %d\n\r", N);
        }

        calculate_averages(cycle_counts, instr_counts, exec_time_us_values, stack_usages, NUM_EXECUTIONS);

        free(input);
        free(original_input);
        free(freq_domain);
    }
    printf("\nBenchmark completed for NMSIS IFFT Q15.\n\r");
}
