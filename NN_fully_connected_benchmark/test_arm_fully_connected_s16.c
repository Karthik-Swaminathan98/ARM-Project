#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "validate.h"

#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include "TestData/fc_int16_slow/test_data.h"
#include "TestData/fully_connected_int16/test_data.h"
#include "../TestData/fully_connected_int16_big/test_data.h"

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

	// Fill stack with a known pattern
	fill_stack_pattern_to_sp();

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

    // Measure stack usage
    uint32_t stack_used_s16 = measure_stack_usage();

	// Calculate cycle count
	uint32_t cycle_count_s16 = end_cycles_s16 - start_cycles_s16;

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    printf("\n\r");
	if (validate_s16(output, output_ref, output_ref_size)) {
		printf("arm_fully_connected_s16 output validation PASSED\n\r");
		printf("Stack Used for arm_fully_connected_s16: %lu bytes\n\r", (unsigned long)stack_used_s16);
		printf("Cycle Count for arm_fully_connected_s16: %lu\n\r", (unsigned long)cycle_count_s16);
	} else {
		printf("arm_fully_connected_s16 output validation FAILED\n\r");
	}
}

void fully_connected_int16_big_arm_fully_connected_s16(void)
{
    int16_t output[FULLY_CONNECTED_INT16_BIG_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = fully_connected_int16_big_biases;
    const int8_t *kernel_data = fully_connected_int16_big_weights;
    const int16_t *input_data = fully_connected_int16_big_input;
    const int16_t *output_ref = fully_connected_int16_big_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_INT16_BIG_DST_SIZE;

    input_dims.n = FULLY_CONNECTED_INT16_BIG_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_INT16_BIG_INPUT_W;
    input_dims.h = FULLY_CONNECTED_INT16_BIG_INPUT_H;
    input_dims.c = FULLY_CONNECTED_INT16_BIG_IN_CH;
    filter_dims.n = FULLY_CONNECTED_INT16_BIG_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_INT16_BIG_OUT_CH;
    filter_dims.h = FULLY_CONNECTED_INT16_BIG_INPUT_H;
    filter_dims.w = FULLY_CONNECTED_INT16_BIG_INPUT_W;
    output_dims.n = FULLY_CONNECTED_INT16_BIG_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_INT16_BIG_OUT_CH;

    fc_params.input_offset = 0;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 0;
    fc_params.activation.min = FULLY_CONNECTED_INT16_BIG_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_INT16_BIG_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FULLY_CONNECTED_INT16_BIG_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_INT16_BIG_OUTPUT_SHIFT;

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
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
    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (validate_s16(output, output_ref, output_ref_size)) {
        printf("arm_fully_connected_s16 (BIG) output validation PASSED\n\r");
        printf("Stack Used for arm_fully_connected_s16 (BIG): %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count for arm_fully_connected_s16 (BIG): %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("arm_fully_connected_s16 (BIG) output validation FAILED\n\r");
    }
}

void fc_int16_slow_arm_fully_connected_s16(void)
{
    int16_t output[FC_INT16_SLOW_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *bias_data = fc_int16_slow_biases;
    const int8_t *kernel_data = fc_int16_slow_weights;
    const int16_t *input_data = fc_int16_slow_input;
    const int16_t *output_ref = fc_int16_slow_output_ref;
    const int32_t output_ref_size = FC_INT16_SLOW_DST_SIZE;

    input_dims.n = FC_INT16_SLOW_INPUT_BATCHES;
    input_dims.w = FC_INT16_SLOW_INPUT_W;
    input_dims.h = FC_INT16_SLOW_INPUT_H;
    input_dims.c = FC_INT16_SLOW_IN_CH;
    filter_dims.n = FC_INT16_SLOW_ACCUMULATION_DEPTH;
    filter_dims.c = FC_INT16_SLOW_OUT_CH;
    filter_dims.h = FC_INT16_SLOW_INPUT_H;
    filter_dims.w = FC_INT16_SLOW_INPUT_W;
    output_dims.n = FC_INT16_SLOW_INPUT_BATCHES;
    output_dims.c = FC_INT16_SLOW_OUT_CH;

    fc_params.input_offset = 0;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 0;
    fc_params.activation.min = FC_INT16_SLOW_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FC_INT16_SLOW_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FC_INT16_SLOW_OUTPUT_MULTIPLIER;
    quant_params.shift = FC_INT16_SLOW_OUTPUT_SHIFT;

    int32_t buf_size = arm_fully_connected_s16_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
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
    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (validate_s16(output, output_ref, output_ref_size)) {
        printf("arm_fully_connected_s16 (SLOW) output validation PASSED\n\r");
        printf("Stack Used for arm_fully_connected_s16 (SLOW): %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count for arm_fully_connected_s16 (SLOW): %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("arm_fully_connected_s16 (SLOW) output validation FAILED\n\r");
    }
}

