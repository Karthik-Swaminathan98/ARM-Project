#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/dw_int16xint8/test_data.h"
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

void dw_int16xint8_arm_depthwise_conv_s16(void)
{
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[DW_INT16XINT8_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = dw_int16xint8_biases;
    const int16_t *input_data = dw_int16xint8_input;
    const int8_t *kernel_data = dw_int16xint8_weights;
    const int16_t *output_ref = dw_int16xint8_output_ref;
    const int32_t output_ref_size = DW_INT16XINT8_DST_SIZE;

    input_dims.n = DW_INT16XINT8_INPUT_BATCHES;
    input_dims.w = DW_INT16XINT8_INPUT_W;
    input_dims.h = DW_INT16XINT8_INPUT_H;
    input_dims.c = DW_INT16XINT8_IN_CH;
    filter_dims.w = DW_INT16XINT8_FILTER_X;
    filter_dims.h = DW_INT16XINT8_FILTER_Y;
    output_dims.w = DW_INT16XINT8_OUTPUT_W;
    output_dims.h = DW_INT16XINT8_OUTPUT_H;
    output_dims.c = DW_INT16XINT8_OUT_CH;

    dw_conv_params.padding.w = DW_INT16XINT8_PAD_X;
    dw_conv_params.padding.h = DW_INT16XINT8_PAD_Y;
    dw_conv_params.stride.w = DW_INT16XINT8_STRIDE_X;
    dw_conv_params.stride.h = DW_INT16XINT8_STRIDE_Y;
    dw_conv_params.dilation.w = DW_INT16XINT8_DILATION_X;
    dw_conv_params.dilation.h = DW_INT16XINT8_DILATION_Y;

    dw_conv_params.ch_mult = DW_INT16XINT8_CH_MULT;

    dw_conv_params.input_offset = DW_INT16XINT8_INPUT_OFFSET;
    dw_conv_params.output_offset = DW_INT16XINT8_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DW_INT16XINT8_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DW_INT16XINT8_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)dw_int16xint8_output_mult;
    quant_params.shift = (int32_t *)dw_int16xint8_output_shift;

    ctx.buf = NULL;
    ctx.size = 0;

	enable_cycle_counter();
	// Measure cycles
    uint32_t start_cycles_dw_s16 = read_cycle_counter();

    arm_depthwise_conv_s16(&ctx,
							&dw_conv_params,
							&quant_params,
							&input_dims,
							input_data,
							&filter_dims,
							dw_int16xint8_weights,
							&bias_dims,
							bias_data,
							&output_dims,
							output);
	// Measure cycles
	uint32_t end_cycles_dw_s16 = read_cycle_counter();

	// Calculate cycle count
	uint32_t cycle_count_dw_s16 = end_cycles_dw_s16 - start_cycles_dw_s16;

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
	if (validate_s16(output, output_ref, output_ref_size)) {
		printf("arm_depthwise_conv_s16 output validation PASSED\n\r");
		printf("Cycle Count for arm_convolve_s16: %lu\n\r", (unsigned long)cycle_count_dw_s16);
	} else {
		printf("arm_depthwise_conv_s16 output validation FAILED\n\r");
	}
    memset(output, 0, sizeof(output));

    int buf_size =
        arm_depthwise_conv_wrapper_s16_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);



    ctx.buf = malloc(buf_size);

    // Measure cycles
    uint32_t start_cycles_dw_s16_w = read_cycle_counter();

    arm_depthwise_conv_wrapper_s16(&ctx,
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
	uint32_t end_cycles_dw_s16_w = read_cycle_counter();

	// Calculate cycle count
	uint32_t cycle_count_dw_s16_w = end_cycles_dw_s16_w - start_cycles_dw_s16_w;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
	if (validate_s16(output, output_ref, output_ref_size)) {
		printf("arm_depthwise_conv_wrapper_s16 output validation PASSED\n\r");
		printf("Cycle Count arm_convolve_wrapper_s16: %lu\n\r", (unsigned long)cycle_count_dw_s16_w);
	} else {
		printf("arm_depthwise_conv_wrapper_s16 output validation FAILED\n\r");
	}

}
