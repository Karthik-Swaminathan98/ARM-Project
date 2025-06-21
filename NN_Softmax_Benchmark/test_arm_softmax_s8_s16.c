#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/softmax_s8_s16/test_data.h"
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

void softmax_s8_s16_arm_softmax_s8_s16(void)
{
    const int32_t num_rows = SOFTMAX_S8_S16_NUM_ROWS;
    const int32_t row_size = SOFTMAX_S8_S16_ROW_SIZE;
    const int32_t mult = SOFTMAX_S8_S16_INPUT_MULT;
    const int32_t shift = SOFTMAX_S8_S16_INPUT_LEFT_SHIFT;
    const int32_t diff_min = SOFTMAX_S8_S16_DIFF_MIN;
    const int8_t *input_data = softmax_s8_s16_input;
    int16_t output[SOFTMAX_S8_S16_DST_SIZE];

	enable_cycle_counter();

	// Fill stack with a known pattern
	fill_stack_pattern_to_sp();

    // Measure cycles
    uint32_t start_cycles_s8_s16 = read_cycle_counter();
    arm_softmax_s8_s16(input_data, num_rows, row_size, mult, shift, diff_min, output);
    uint32_t end_cycles_s8_s16 = read_cycle_counter();

    // Measure stack usage
    uint32_t stack_used_s8_s16 = measure_stack_usage();

    // Calculate cycle count
    uint32_t cycle_count_s8_s16 = end_cycles_s8_s16 - start_cycles_s8_s16;
	if (validate_s16(output, softmax_s8_s16_output_ref, SOFTMAX_S8_S16_DST_SIZE)) {
		printf("arm_softmax_s8_s16 output validation PASSED\n\r");
		printf("Stack Used for arm_softmax_s8_s16: %lu bytes\n\r", (unsigned long)stack_used_s8_s16);
		printf("Cycle Count for arm_softmax_s8_s16: %lu\n\r", (unsigned long)cycle_count_s8_s16);
	} else {
		printf("arm_softmax_s8_s16 output validation FAILED\n\r");
	}

}
