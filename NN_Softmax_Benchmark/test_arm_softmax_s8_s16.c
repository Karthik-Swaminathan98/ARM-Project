#include "main.h"
#include "../Include/softmax_s8_s16/test_data.h"

#define REPEAT_NUM (2)

RAM_FUNC void softmax_s8_s16_arm_softmax_s8_s16(void)
{
    const int32_t num_rows = SOFTMAX_S8_S16_NUM_ROWS;
    const int32_t row_size = SOFTMAX_S8_S16_ROW_SIZE;
    const int32_t mult = SOFTMAX_S8_S16_INPUT_MULT;
    const int32_t shift = SOFTMAX_S8_S16_INPUT_LEFT_SHIFT;
    const int32_t diff_min = SOFTMAX_S8_S16_DIFF_MIN;
    const int8_t *input_data = softmax_s8_s16_input;
    int16_t output[SOFTMAX_S8_S16_DST_SIZE];

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles = read_cycle_counter();

    arm_softmax_s8_s16(input_data, num_rows, row_size, mult, shift, diff_min, output);

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

	if (validate_s16(output, softmax_s8_s16_output_ref, SOFTMAX_S8_S16_DST_SIZE)) {
		printf("arm_softmax_s8_s16 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
	} else {
		printf("arm_softmax_s8_s16 output validation FAILED\n\r");
	}

}
