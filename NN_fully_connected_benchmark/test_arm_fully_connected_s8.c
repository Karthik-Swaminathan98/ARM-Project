#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/fc_per_ch/test_data.h"
#include "../Include/fully_connected/test_data.h"
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

void fully_connected_arm_fully_connected_s8(void)
{
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[FULLY_CONNECTED_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fully_connected_biases;
    const int8_t *kernel_data = fully_connected_weights;
    const int8_t *input_data = fully_connected_input;
    const int8_t *output_ref = fully_connected_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_DST_SIZE;

    input_dims.n = FULLY_CONNECTED_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_INPUT_W;
    input_dims.h = FULLY_CONNECTED_INPUT_H;
    input_dims.c = FULLY_CONNECTED_IN_CH;
    filter_dims.n = FULLY_CONNECTED_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_OUT_CH;
    output_dims.n = FULLY_CONNECTED_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_OUT_CH;

    fc_params.input_offset = FULLY_CONNECTED_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FULLY_CONNECTED_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_OUTPUT_SHIFT;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    TEST_ASSERT_EQUAL(expected,
                      arm_vector_sum_s8(buf,
                                        filter_dims.n,
                                        output_dims.c,
                                        kernel_data,
                                        fc_params.input_offset,
                                        fc_params.filter_offset,
                                        bias_data));
#endif

	enable_cycle_counter();
    // Measure cycles
    uint32_t start_cycles_s8 = read_cycle_counter();

    arm_cmsis_nn_status result = arm_fully_connected_s8(&ctx,
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
    uint32_t end_cycles_s8 = read_cycle_counter();

    // Calculate cycle count
    uint32_t cycle_count_s8 = end_cycles_s8 - start_cycles_s8;
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
	if (validate(output, output_ref, output_ref_size)) {
		printf("arm_fully_connected_s8 output validation PASSED\n\r");
		printf("Cycle Count for arm_fully_connected_s8: %lu\n\r", (unsigned long)cycle_count_s8);
	} else {
		printf("arm_fully_connected_s8 output validation FAILED\n\r");
	}
}
