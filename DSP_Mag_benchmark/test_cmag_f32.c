#include "main.h"

RAM_FUNC void generate_sine_wave_f32(float32_t* input, int N, float signal_freq, float sampling_freq) {
    for (int i = 0; i < N; i++) {
        float value = sinf(2 * M_PI * signal_freq * i / sampling_freq);
        input[2 * i] = value;  // Real part
        input[2 * i + 1] = 0;  // Imaginary part
    }
}

RAM_FUNC void benchmark_cmag_f32() {
    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        float32_t* input = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* original_input = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* magnitude = (float32_t*)malloc(N * sizeof(float32_t));
        float32_t* magnitude_reference = (float32_t*)malloc(N * sizeof(float32_t));
        if (!input || !original_input || !magnitude || !magnitude_reference) {
            printf("Memory allocation failed for FFT size N = %d\n\r", N);
            //return -1;
        }

        generate_sine_wave_f32(original_input, N, SINE_FREQ, SAMPLING_FREQ);

        // Initialize FFT instance
        arm_cfft_instance_f32 fft_instance;
        if (arm_cfft_init_f32(&fft_instance, N) != ARM_MATH_SUCCESS) {
            printf("FFT init failed for N = %d\n", N);
            free(input); free(original_input); free(magnitude); free(magnitude_reference);
            continue;
        }

        uint32_t cycle_counts[NUM_EXECUTIONS];
        uint32_t instr_counts[NUM_EXECUTIONS];
        uint32_t stack_usages[NUM_EXECUTIONS];
        float exec_time_us_values[NUM_EXECUTIONS];

        int magnitudes_consistent = 1;
        printf("\nFFT Size: %d\n", N);

        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
        	memcpy(input, original_input, 2 * N * sizeof(float32_t));
        	arm_cfft_f32(&fft_instance, input, 0, 1);

            fill_stack_pattern_to_sp();
            enable_cycle_counter();
            uint32_t start_cycles = read_cycle_counter();

            arm_cmplx_mag_f32(input, magnitude, N);

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
//            printf("\n\r");
//            printf("Execution %d: Cycle Count = %lu, Estimated Instructions = %lu, Time = %.2f us, Stack Used = %lu bytes\n\r",
//                   execution + 1, cycle_counts[execution], instr_counts[execution],
//                   exec_time_us_values[execution], stack_usages[execution]);
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

    printf("\nBenchmark completed for ARM Cmag F32.\n\r");
    //return 0;
}
