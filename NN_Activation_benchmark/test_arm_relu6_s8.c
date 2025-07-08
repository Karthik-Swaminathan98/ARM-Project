#include "main.h"

RAM_FUNC void relu6_arm_relu6_s8(void) {
    const uint16_t input_sizes[] = {512, 1024, 2048};
    const int num_sizes = sizeof(input_sizes) / sizeof(input_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        uint16_t size = input_sizes[i];

        // Allocate and initialize input buffer
        int8_t *input_data = (int8_t *)malloc(size * sizeof(int8_t));
        if (!input_data) {
            printf("Memory allocation failed for size %u\n\r", size);
            continue;
        }

        generate_rand_s8(input_data, size);

        fill_stack_pattern_to_sp();
        enable_cycle_counter();
        uint32_t start_cycles = read_cycle_counter();

        arm_relu6_s8(input_data, size);

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

        printf("Input Size: %u\n\r", size);
        printf("Cycle Count for arm_relu6_s8: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);

        free(input_data);
    }
}
