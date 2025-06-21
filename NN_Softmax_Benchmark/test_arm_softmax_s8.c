#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/softmax/test_data.h"
#include "validate.h"

#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#define REPEAT_NUM (2)

static void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; // Enable DWT
    DWT->CYCCNT = 0;                                // Reset cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;            // Enable cycle counter
}

static uint32_t read_cycle_counter() {
    return DWT->CYCCNT;
}

extern uint32_t __StackLimit;

static void fill_stack_pattern_to_sp() {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));

    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) {
        *p++ = 0xAAAAAAAA;
    }
}

static uint32_t measure_stack_usage() {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));

    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) {
        if (*p != 0xAAAAAAAA) {
            break;
        }
        p++;
    }

    return ((uint32_t)sp - (uint32_t)p); // Stack usage in bytes
}

void softmax_arm_softmax_s8(void)
{
    const int32_t num_rows = SOFTMAX_NUM_ROWS;
    const int32_t row_size = SOFTMAX_ROW_SIZE;
    const int32_t mult = SOFTMAX_INPUT_MULT;
    const int32_t shift = SOFTMAX_INPUT_LEFT_SHIFT;
    const int32_t diff_min = SOFTMAX_DIFF_MIN;
    const int8_t *input_data = softmax_input;
    int8_t output[SOFTMAX_DST_SIZE];

    enable_cycle_counter();

	// Fill stack with a known pattern
	fill_stack_pattern_to_sp();

	// Measure cycles
	uint32_t start_cycles_s8 = read_cycle_counter();
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);
    // Measure cycles
    uint32_t end_cycles_s8 = read_cycle_counter();

    // Measure stack usage
    uint32_t stack_used_s8 = measure_stack_usage();

    // Calculate cycle count
    uint32_t cycle_count_s8 = end_cycles_s8 - start_cycles_s8;
    printf("\n\r");
    if (validate(output, softmax_output_ref, SOFTMAX_DST_SIZE)) {
		printf("arm_softmax_s8 output validation PASSED\n\r");
		printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used_s8);
		printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count_s8);
	} else {
		printf("arm_avgparm_softmax_s8ool_s8 output validation FAILED\n\r");
	}

}
void softmax_invalid_diff_min_arm_softmax_s8(void)
{
    const int32_t num_rows = SOFTMAX_NUM_ROWS;
    const int32_t row_size = SOFTMAX_ROW_SIZE;
    const int32_t mult = SOFTMAX_INPUT_MULT;
    const int32_t shift = SOFTMAX_INPUT_LEFT_SHIFT;
    const int32_t diff_min = 0x7FFFFFFF;
    const int8_t *input_data = softmax_input;
    int8_t output[SOFTMAX_DST_SIZE];

    int8_t *softmax_expect_invalid_output = malloc(SOFTMAX_DST_SIZE);
    for (int i = 0; i < SOFTMAX_DST_SIZE; i++)
    {
        softmax_expect_invalid_output[i] = -128;
    }

    enable_cycle_counter();

    // Fill stack with a known pattern
    fill_stack_pattern_to_sp();

    // Measure cycles
    uint32_t start_cycles = read_cycle_counter();
    arm_softmax_s8(input_data, num_rows, row_size, mult, shift, diff_min, output);
    uint32_t end_cycles = read_cycle_counter();

    // Measure stack usage
    uint32_t stack_used = measure_stack_usage();

    // Calculate cycle count
    uint32_t cycle_count = end_cycles - start_cycles;
    printf("\n\r");
    if (validate(output, softmax_expect_invalid_output, SOFTMAX_DST_SIZE)) {
        printf("arm_softmax_s8 with invalid diff_min output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("arm_softmax_s8 with invalid diff_min output validation FAILED\n\r");
    }

    free(softmax_expect_invalid_output);
}

