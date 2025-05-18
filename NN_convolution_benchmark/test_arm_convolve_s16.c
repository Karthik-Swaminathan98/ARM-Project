#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/int16xint8/test_data.h"
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

void basic_arm_convolve_s16(void)
{
	//const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
	int16_t output[INT16XINT8_DST_SIZE] = {0};

	cmsis_nn_context ctx;
	cmsis_nn_conv_params conv_params;
	cmsis_nn_per_channel_quant_params quant_params;
	cmsis_nn_dims input_dims;
	cmsis_nn_dims filter_dims;
	cmsis_nn_dims bias_dims;
	cmsis_nn_dims output_dims;

	const int64_t *int64_bias_data = int16xint8_biases;
	const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
	const int8_t *kernel_data = int16xint8_weights;
	const int16_t *input_data = int16xint8_input;
	const int16_t *output_ref = int16xint8_output_ref;
	const int32_t output_ref_size = INT16XINT8_DST_SIZE;

	input_dims.n = INT16XINT8_INPUT_BATCHES;
	input_dims.w = INT16XINT8_INPUT_W;
	input_dims.h = INT16XINT8_INPUT_H;
	input_dims.c = INT16XINT8_IN_CH;
	filter_dims.w = INT16XINT8_FILTER_X;
	filter_dims.h = INT16XINT8_FILTER_Y;
	output_dims.w = INT16XINT8_OUTPUT_W;
	output_dims.h = INT16XINT8_OUTPUT_H;
	output_dims.c = INT16XINT8_OUT_CH;

	conv_params.padding.w = INT16XINT8_PAD_X;
	conv_params.padding.h = INT16XINT8_PAD_Y;
	conv_params.stride.w = INT16XINT8_STRIDE_X;
	conv_params.stride.h = INT16XINT8_STRIDE_Y;
	conv_params.dilation.w = INT16XINT8_DILATION_X;
	conv_params.dilation.h = INT16XINT8_DILATION_Y;

	conv_params.input_offset = 0;
	conv_params.output_offset = 0;
	conv_params.activation.min = INT16XINT8_OUT_ACTIVATION_MIN;
	conv_params.activation.max = INT16XINT8_OUT_ACTIVATION_MAX;
	quant_params.multiplier = (int32_t *)int16xint8_output_mult;
	quant_params.shift = (int32_t *)int16xint8_output_shift;

	int buf_size = arm_convolve_s16_get_buffer_size(&input_dims, &filter_dims);
	ctx.buf = malloc(buf_size);

	enable_cycle_counter();
	// Measure cycles
    uint32_t start_cycles_s16 = read_cycle_counter();
	arm_convolve_s16(&ctx,
					  &conv_params,
					  &quant_params,
					  &input_dims,
					  input_data,
					  &filter_dims,
					  kernel_data,
					  &bias_dims,
					  &bias_data,
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
		printf("arm_convolve_s16 output validation PASSED\n\r");
		printf("Cycle Count for arm_convolve_s16: %lu\n\r", (unsigned long)cycle_count_s16);
	} else {
		printf("arm_convolve_s16 output validation FAILED\n\r");
	}
	memset(output, 0, sizeof(output));

	buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
	ctx.buf = malloc(buf_size);

    // Measure cycles
    uint32_t start_cycles_s16_w = read_cycle_counter();

	arm_convolve_wrapper_s16(&ctx,
							  &conv_params,
							  &quant_params,
							  &input_dims,
							  input_data,
							  &filter_dims,
							  kernel_data,
							  &bias_dims,
							  &bias_data,
							  &output_dims,
							  output);
	// Measure cycles
	uint32_t end_cycles_s16_w = read_cycle_counter();

	// Calculate cycle count
	uint32_t cycle_count_s16_w = end_cycles_s16_w - start_cycles_s16_w;
	if (ctx.buf)
	{
		memset(ctx.buf, 0, buf_size);
		free(ctx.buf);
	}

	if (validate_s16(output, output_ref, output_ref_size)) {
		printf("arm_convolve_wrapper_s16 output validation PASSED\n\r");
		printf("Cycle Count arm_convolve_wrapper_s16: %lu\n\r", (unsigned long)cycle_count_s16_w);
	} else {
		printf("arm_convolve_wrapper_s16 output validation FAILED\n\r");
	}

}
