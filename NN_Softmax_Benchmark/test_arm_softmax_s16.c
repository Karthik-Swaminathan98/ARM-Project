#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/softmax_s16/test_data.h"
#include "../Include/softmax_s16/exp_lut_data.h"
#include "../Include/softmax_s16/one_by_one_lut_data.h"
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

void softmax_s16_arm_softmax_s16(void)
{
    const int32_t num_rows = SOFTMAX_S16_NUM_ROWS;
    const int32_t row_size = SOFTMAX_S16_ROW_SIZE;
    const int32_t mult = SOFTMAX_S16_INPUT_MULT;
    const int32_t shift = SOFTMAX_S16_INPUT_LEFT_SHIFT;
    const int16_t *input_data = softmax_s16_input;
    const cmsis_nn_softmax_lut_s16 softmax_params = {.exp_lut = softmax_s16_exp_lut,
                                                     .one_by_one_lut = softmax_s16_one_by_one_lut};
    int16_t output[SOFTMAX_S16_DST_SIZE];

    enable_cycle_counter();
	// Measure cycles
	uint32_t start_cycles_s16 = read_cycle_counter();

    arm_softmax_s16(input_data, num_rows, row_size, mult, shift, &softmax_params, output);

	// Measure cycles
	uint32_t end_cycles_s16 = read_cycle_counter();

	// Calculate cycle count
	uint32_t cycle_count_s16 = end_cycles_s16 - start_cycles_s16;

    if (validate_s16(output, softmax_s16_output_ref, SOFTMAX_S16_DST_SIZE)) {
		printf("arm_softmax_s16 output validation PASSED\n\r");
		printf("Cycle for arm_softmax_s16: %lu\n\r", (unsigned long)cycle_count_s16);
	} else {
		printf("arm_softmax_s16 output validation FAILED\n\r");
	}
}
