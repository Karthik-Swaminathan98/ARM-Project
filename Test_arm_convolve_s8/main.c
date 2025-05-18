#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "test_data.h"
#include "validate.h"
int main(void)
{
    cy_rslt_t result;

    // Initialize the device and board peripherals
    result = cybsp_init();
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    // Enable global interrupts
    __enable_irq();

    // Initialize retarget-io to use the debug UART port
    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                                    CYBSP_DEBUG_UART_CTS, CYBSP_DEBUG_UART_RTS, CY_RETARGET_IO_BAUDRATE);

    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }
    const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
	int8_t output[BASIC_DST_SIZE] = {0};

	cmsis_nn_context ctx;
	cmsis_nn_conv_params conv_params;
	cmsis_nn_per_channel_quant_params quant_params;
	cmsis_nn_dims input_dims;
	cmsis_nn_dims filter_dims;
	cmsis_nn_dims bias_dims;
	cmsis_nn_dims output_dims;

	const int32_t *bias_data = basic_biases;
	const int8_t *kernel_data = basic_weights;
	const int8_t *input_data = basic_input;
	const int8_t *output_ref = basic_output_ref;
	const int32_t output_ref_size = BASIC_DST_SIZE;

	input_dims.n = BASIC_INPUT_BATCHES;
	input_dims.w = BASIC_INPUT_W;
	input_dims.h = BASIC_INPUT_H;
	input_dims.c = BASIC_IN_CH;
	filter_dims.w = BASIC_FILTER_X;
	filter_dims.h = BASIC_FILTER_Y;
	filter_dims.c = BASIC_IN_CH;
	output_dims.w = BASIC_OUTPUT_W;
	output_dims.h = BASIC_OUTPUT_H;
	output_dims.c = BASIC_OUT_CH;

	conv_params.padding.w = BASIC_PAD_X;
	conv_params.padding.h = BASIC_PAD_Y;
	conv_params.stride.w = BASIC_STRIDE_X;
	conv_params.stride.h = BASIC_STRIDE_Y;
	conv_params.dilation.w = BASIC_DILATION_X;
	conv_params.dilation.h = BASIC_DILATION_Y;

	conv_params.input_offset = BASIC_INPUT_OFFSET;
	conv_params.output_offset = BASIC_OUTPUT_OFFSET;
	conv_params.activation.min = BASIC_OUT_ACTIVATION_MIN;
	conv_params.activation.max = BASIC_OUT_ACTIVATION_MAX;
	quant_params.multiplier = (int32_t *)basic_output_mult;
	quant_params.shift = (int32_t *)basic_output_shift;

	int32_t buf_size = arm_convolve_s8_get_buffer_size(&input_dims, &filter_dims);
	ctx.buf = malloc(buf_size);
	ctx.size = 0;

	arm_cmsis_nn_status results = arm_convolve_s8(&ctx,
												 &conv_params,
												 &quant_params,
												 &input_dims,
												 input_data,
												 &filter_dims,
												 kernel_data,
												 &bias_dims,
												 bias_data,
												 NULL,
												 &output_dims,
												 output);

	if (ctx.buf)
	{
		// The caller is responsible to clear the scratch buffers for security reasons if applicable.
		memset(ctx.buf, 0, buf_size);
		free(ctx.buf);
	}
//	TEST_ASSERT_EQUAL(expected, results);
//	TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
	printf("\n\r");
	if (results == expected) {
		printf("arm_convolve_s8 returned expected status: SUCCESS\n\r");
	} else {
		printf("arm_convolve_s8 FAILED! Status: %d\n\r", results);
	}

	if (validate(output, output_ref, output_ref_size)) {
		printf("arm_convolve_s8 output validation PASSED\n\r");
	} else {
		printf("arm_convolve_s8 output validation FAILED\n\r");
	}

	// Print output tensor
	printf("Output of arm_convolve_s8:\n\r");
	for (int i = 0; i < output_ref_size; i++) {
	    printf("%d \n\r", output[i]);
	}
	printf("\n\r");

	memset(output, 0, sizeof(output));

	buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
	ctx.buf = malloc(buf_size);
	ctx.size = 0;

	results = arm_convolve_wrapper_s8(&ctx,
									 &conv_params,
									 &quant_params,
									 &input_dims,
									 input_data,
									 &filter_dims,
									 kernel_data,
									 &bias_dims,
									 bias_data,
									 &output_dims,
									 output);

	if (ctx.buf)
	{
		memset(ctx.buf, 0, buf_size);
		free(ctx.buf);
	}
//	TEST_ASSERT_EQUAL(expected, results);
//	TEST_ASSERT_TRUE(validate(output, output_ref, output_ref_size));
	if (results == expected) {
		printf("arm_convolve_wrapper_s8 returned expected status: SUCCESS\n\r");
	} else {
		printf("arm_convolve_wrapper_s8 FAILED! Status: %d\n\r", results);
	}

	if (validate(output, output_ref, output_ref_size)) {
		printf("arm_convolve_wrapper_s8 output validation PASSED\n\r");
	} else {
		printf("arm_convolve_wrapper_s8 output validation FAILED\n\r");
	}

	printf("Output of arm_convolve_wrapper_s8:\n\r");
	for (int i = 0; i < output_ref_size; i++) {
	    printf("%d \n\r", output[i]);
	}
	return 0;

}
