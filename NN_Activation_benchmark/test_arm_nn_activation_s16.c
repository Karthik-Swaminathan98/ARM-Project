#include "main.h"

#define LEFT_SHIFT 8
#define ACTIVATION_FUNC ARM_SIGMOID // You can change to ARM_TANH if needed

RAM_FUNC void activ_arm_nn_activation_s16(void) {
    const int input_sizes[] = {512, 1024, 2048};
    const int num_sizes = sizeof(input_sizes) / sizeof(input_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int size = input_sizes[i];

        // Allocate input and output buffers
        int16_t *input = (int16_t *)malloc(size * sizeof(int16_t));
        int16_t *output = (int16_t *)malloc(size * sizeof(int16_t));

        if (!input || !output) {
            printf("Memory allocation failed for input size %d\n\r", size);
            free(input);
            free(output);
            continue;
        }

        generate_rand_s16(input, size);
        memset(output, 0, size * sizeof(int16_t));

        enable_cycle_counter();
        fill_stack_pattern_to_sp();
        uint32_t start_cycles = read_cycle_counter();

        arm_nn_activation_s16(input, output, size, LEFT_SHIFT, ACTIVATION_FUNC);

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

        printf("Input Size: %d\n\r", size);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);

        free(input);
        free(output);
    }
}
