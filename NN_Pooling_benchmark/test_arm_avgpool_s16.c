#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../Include/avgpooling_int16/test_data.h"
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

void avgpooling_int16_arm_avgpool_s16(void)
{
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int16_t output[AVGPOOLING_INT16_OUTPUT_C * AVGPOOLING_INT16_OUTPUT_W * AVGPOOLING_INT16_OUTPUT_H *
                   AVGPOOLING_INT16_BATCH_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_data = avgpooling_int16_input_tensor;

    input_dims.n = AVGPOOLING_INT16_BATCH_SIZE;
    input_dims.w = AVGPOOLING_INT16_INPUT_W;
    input_dims.h = AVGPOOLING_INT16_INPUT_H;
    input_dims.c = AVGPOOLING_INT16_INPUT_C;
    filter_dims.w = AVGPOOLING_INT16_FILTER_W;
    filter_dims.h = AVGPOOLING_INT16_FILTER_H;
    output_dims.w = AVGPOOLING_INT16_OUTPUT_W;
    output_dims.h = AVGPOOLING_INT16_OUTPUT_H;
    output_dims.c = AVGPOOLING_INT16_INPUT_C;

    pool_params.padding.w = AVGPOOLING_INT16_PADDING_W;
    pool_params.padding.h = AVGPOOLING_INT16_PADDING_H;
    pool_params.stride.w = AVGPOOLING_INT16_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_INT16_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_INT16_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_INT16_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_OUTPUT_W, AVGPOOLING_INT16_INPUT_C);
    ctx.buf = malloc(ctx.size);

    enable_cycle_counter();
	// Measure cycles
	uint32_t start_cycles_s16 = read_cycle_counter();
    arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);
	// Measure cycles
	uint32_t end_cycles_s16 = read_cycle_counter();

	// Calculate cycle count
	uint32_t cycle_count_s16 = end_cycles_s16 - start_cycles_s16;
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    if (validate_s16(output,
            avgpooling_int16_output,
            AVGPOOLING_INT16_OUTPUT_C * AVGPOOLING_INT16_OUTPUT_W * AVGPOOLING_INT16_OUTPUT_H *
                AVGPOOLING_INT16_BATCH_SIZE)) {
		printf("arm_avgpool_s16 output validation PASSED\n\r");
		printf("Cycle for arm_avgpool_s16: %lu\n\r", (unsigned long)cycle_count_s16);
	} else {
		printf("arm_avgpool_s16 output validation FAILED\n\r");
	}
}
