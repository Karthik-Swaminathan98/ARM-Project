#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/depthwise_2/test_data.h"
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

void depthwise_2_arm_depthwise_conv_s8(void)
{
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[DEPTHWISE_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_2_DST_SIZE;
    const int32_t *bias_data = depthwise_2_biases;
    const int8_t *kernel_data = depthwise_2_weights;
    const int8_t *input_data = depthwise_2_input;

    input_dims.n = DEPTHWISE_2_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_2_INPUT_W;
    input_dims.h = DEPTHWISE_2_INPUT_H;
    input_dims.c = DEPTHWISE_2_IN_CH;
    filter_dims.w = DEPTHWISE_2_FILTER_X;
    filter_dims.h = DEPTHWISE_2_FILTER_Y;
    output_dims.w = DEPTHWISE_2_OUTPUT_W;
    output_dims.h = DEPTHWISE_2_OUTPUT_H;
    output_dims.c = DEPTHWISE_2_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_2_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_2_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_2_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_2_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_2_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_2_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_2_CH_MULT;

    dw_conv_params.input_offset = DEPTHWISE_2_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_2_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_2_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_2_output_mult;
    quant_params.shift = (int32_t *)depthwise_2_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

	enable_cycle_counter();
    // Measure cycles
    uint32_t start_cycles_dw_s8 = read_cycle_counter();

    arm_depthwise_conv_s8(&ctx,
						   &dw_conv_params,
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
    uint32_t end_cycles_dw_s8 = read_cycle_counter();

    // Calculate cycle count
    uint32_t cycle_count_dw_s8 = end_cycles_dw_s8 - start_cycles_dw_s8;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    if (validate(output, depthwise_2_output_ref, output_ref_size)) {
		printf("arm_depthwise_conv_s8 output validation PASSED\n\r");
		printf("Cycle Count for arm_depthwise_conv_s8: %lu\n\r", (unsigned long)cycle_count_dw_s8);
	} else {
		printf("arm_depthwise_conv_s8 output validation FAILED\n\r");
	}
    memset(output, 0, sizeof(output));

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);


    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    // Measure cycles
    uint32_t start_cycles_dw_s8_w = read_cycle_counter();
    arm_depthwise_conv_wrapper_s8(&ctx,
								   &dw_conv_params,
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
	uint32_t end_cycles_dw_s8_w = read_cycle_counter();

	// Calculate cycle count
	uint32_t cycle_count_dw_s8_w = end_cycles_dw_s8_w - start_cycles_dw_s8_w;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
	if (validate(output, depthwise_2_output_ref, output_ref_size)) {
		printf("arm_depthwise_conv_wrapper_s8 output validation PASSED\n\r");
		printf("Cycle Count arm_depthwise_conv_wrapper_s8: %lu\n\r", (unsigned long)cycle_count_dw_s8_w);
	} else {
		printf("arm_depthwise_conv_wrapper_s8 output validation FAILED\n\r");
	}
}
