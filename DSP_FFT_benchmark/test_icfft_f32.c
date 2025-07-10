#include "main.h"

RAM_FUNC void benchmark_ifft_f32() {
    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        float32_t* input = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* freq_domain = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* time_domain_ref = (float32_t*)malloc(2 * N * sizeof(float32_t));
        if (!input || !freq_domain || !time_domain_ref) {
            printf("Memory allocation failed for FFT size N = %d\n\r", N);
            continue;
        }

        // Generate original time domain signal
        generate_sine_wave_f32(time_domain_ref, N, SINE_FREQ, SAMPLING_FREQ);

        // Forward FFT to create freq_domain input for IFFT
        memcpy(input, time_domain_ref, 2 * N * sizeof(float32_t));
        arm_cfft_instance_f32 fft_instance_fwd;
        arm_cfft_init_f32(&fft_instance_fwd, N);
        arm_cfft_f32(&fft_instance_fwd, input, 0, 1);
        memcpy(freq_domain, input, 2 * N * sizeof(float32_t));  // Save freq domain result

        arm_cfft_instance_f32 fft_instance_inv;
        if (arm_cfft_init_f32(&fft_instance_inv, N) != ARM_MATH_SUCCESS) {
            printf("IFFT init failed for N = %d\n", N);
            free(input); free(freq_domain); free(time_domain_ref);
            continue;
        }

        uint32_t cycle_counts[NUM_EXECUTIONS];
        uint32_t instr_counts[NUM_EXECUTIONS];
        uint32_t stack_usages[NUM_EXECUTIONS];
        float exec_time_us_values[NUM_EXECUTIONS];

        int signals_consistent = 1;
        printf("\nIFFT Size: %d\n", N);

        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            // Use freq_domain as input to IFFT
            memcpy(input, freq_domain, 2 * N * sizeof(float32_t));

            fill_stack_pattern_to_sp();
            enable_cycle_counter();
            uint32_t start_cycles = read_cycle_counter();

            // The core: **inverse** FFT (ifftFlag = 1)
            arm_cfft_f32(&fft_instance_inv, input, 1, 1);

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
            float time_us = time_sec * 1e6f;;

            cycle_counts[execution] = cycle_count;
            instr_counts[execution] = instr_est;
            stack_usages[execution] = stack_used;
            exec_time_us_values[execution] = time_us;

            // Consistency check: compare time-domain output to reference
            if (execution == 0) {
                // You may print a few output samples if you want:
                // for (int i = 0; i < N; i++) printf("%d: %.4f (ref: %.4f)\n\r", i, input[2*i], time_domain_ref[2*i]);
            } else {
                for (int i = 0; i < N; i++) {
                    if (fabsf(input[2 * i] - time_domain_ref[2 * i]) > 1e-3) { // Only real part (sine wave)
                        signals_consistent = 0;
                        break;
                    }
                }
            }
        }

        if (signals_consistent) {
            printf("All IFFT outputs are consistent across executions for FFT size N = %d\n\r", N);
        } else {
            printf("Inconsistent IFFT outputs detected for FFT size N = %d\n\r", N);
        }

        calculate_averages(cycle_counts, instr_counts, exec_time_us_values, stack_usages, NUM_EXECUTIONS);

        free(input);
        free(freq_domain);
        free(time_domain_ref);
    }

    printf("\nBenchmark completed for ARM IFFT F32.\n\r");
}
