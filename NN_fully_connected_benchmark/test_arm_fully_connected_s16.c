#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/fc_int16_slow/test_data.h"
#include "../Include/fully_connected_int16/test_data.h"
#include "validate.h"

#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; // Enable DWT
    DWT->CYCCNT = 0;                                // Reset cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;            // Enable cycle counter
}

static uint32_t read_cycle_counter() {
    return DWT->CYCCNT;
}

void fully_connected_int16_arm_fully_connected_s16(void)
{
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[FULLY_CONNECTED_INT16_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = fully_connected_int16_biases;
    const int8_t *kernel_data = fully_connected_int16_weights;
    const int16_t *input_data = fully_connected_int16_input;
    const int16_t *output_ref = fully_connected_int16_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_INT16_DST_SIZE;

    input_dims.n = FULLY_CONNECTED_INT16_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_INT16_INPUT_W;
    input_dims.h = FULLY_CONNECTED_INT16_INPUT_H;
    input_dims.c = FULLY_CONNECTED_INT16_IN_CH;
    filter_dims.n = FULLY_CONNECTED_INT16_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_INT16_OUT_CH;
    filter_dims.h = FULLY_CONNECTED_INT16_INPUT_H;
    filter_dims.w = FULLY_CONNECTED_INT16_INPUT_W;
    output_dims.n = FULLY_CONNECTED_INT16_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_INT16_OUT_CH;

    fc_params.input_offset = 0;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 0;
    fc_params.activation.min = FULLY_CONNECTED_INT16_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_INT16_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FULLY_CONNECTED_INT16_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_INT16_OUTPUT_SHIFT;

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    enable_cycle_counter();
	// Measure cycles
	uint32_t start_cycles_s16 = read_cycle_counter();
    arm_fully_connected_s16(&ctx,
							 &fc_params,
							 &quant_params,
							 &input_dims,
							 input_data,
							 &filter_dims,
							 kernel_data,
							 &bias_dims,
							 bias_data,
							 &output_dims,
							 output);
	// Measure cycles
	uint32_t end_cycles_s16 = read_cycle_counter();

	// Calculate cycle count
	uint32_t cycle_count_s16 = end_cycles_s16 - start_cycles_s16;

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
	if (validate_s16(output, output_ref, output_ref_size)) {
		printf("arm_fully_connected_s16 output validation PASSED\n\r");
		printf("Cycle Count for arm_fully_connected_s16: %lu\n\r", (unsigned long)cycle_count_s16);
	} else {
		printf("arm_fully_connected_s16 output validation FAILED\n\r");
	}
}
