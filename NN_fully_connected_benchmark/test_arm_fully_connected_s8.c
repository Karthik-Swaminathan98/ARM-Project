#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../TestData/fully_connected_mve_0/test_data.h"
#include "TestData/fc_per_ch/test_data.h"
#include "TestData/fully_connected/test_data.h"
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

	// Fill stack with a known pattern
	fill_stack_pattern_to_sp();

    // Measure cycles
    uint32_t start_cycles_s8 = read_cycle_counter();

    arm_fully_connected_s8(&ctx,
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

    // Measure stack usage
    uint32_t stack_used_s8 = measure_stack_usage();

    // Calculate cycle count
    uint32_t cycle_count_s8 = end_cycles_s8 - start_cycles_s8;
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    printf("\n\r");
	if (validate(output, output_ref, output_ref_size)) {
		printf("arm_fully_connected_s8 output validation PASSED\n\r");
		printf("Stack Used for arm_fully_connected_s8: %lu bytes\n\r", (unsigned long)stack_used_s8);
		printf("Cycle Count for arm_fully_connected_s8: %lu\n\r", (unsigned long)cycle_count_s8);
	} else {
		printf("arm_fully_connected_s8 output validation FAILED\n\r");
	}
}

void fc_per_ch_arm_fully_connected_s8(void)
{
    int8_t output[FC_PER_CH_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fc_per_ch_biases;
    const int8_t *kernel_data = fc_per_ch_weights;
    const int8_t *input_data = fc_per_ch_input;
    const int8_t *output_ref = fc_per_ch_output_ref;
    const int32_t output_ref_size = FC_PER_CH_DST_SIZE;

    input_dims.n = FC_PER_CH_INPUT_BATCHES;
    input_dims.w = FC_PER_CH_INPUT_W;
    input_dims.h = FC_PER_CH_INPUT_H;
    input_dims.c = FC_PER_CH_IN_CH;
    filter_dims.n = FC_PER_CH_ACCUMULATION_DEPTH;
    filter_dims.c = FC_PER_CH_OUT_CH;
    output_dims.n = FC_PER_CH_INPUT_BATCHES;
    output_dims.c = FC_PER_CH_OUT_CH;

    fc_params.input_offset = FC_PER_CH_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FC_PER_CH_OUTPUT_OFFSET;
    fc_params.activation.min = FC_PER_CH_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FC_PER_CH_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t *)fc_per_ch_output_mult;
    quant_params.shift = (int32_t *)fc_per_ch_output_shift;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(buf,
                      filter_dims.n,
                      output_dims.c,
                      kernel_data,
                      fc_params.input_offset,
                      fc_params.filter_offset,
                      bias_data);
#endif

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles = read_cycle_counter();

    arm_fully_connected_per_channel_s8(&ctx,
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

    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
        printf("fc_per_ch_arm_fully_connected_s8 output validation PASSED (per-channel)\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("fc_per_ch_arm_fully_connected_s8 output validation FAILED (per-channel)\n\r");
    }

    // Wrapper function test
    cmsis_nn_quant_params generic_quant_params;
    generic_quant_params.multiplier = quant_params.multiplier;
    generic_quant_params.shift = quant_params.shift;
    generic_quant_params.is_per_channel = 1;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles_wrap = read_cycle_counter();

    arm_fully_connected_wrapper_s8(&ctx,
                                   &fc_params,
                                   &generic_quant_params,
                                   &input_dims,
                                   input_data,
                                   &filter_dims,
                                   kernel_data,
                                   &bias_dims,
                                   bias_data,
                                   &output_dims,
                                   output);

    uint32_t end_cycles_wrap = read_cycle_counter();
    uint32_t stack_used_wrap = measure_stack_usage();
    uint32_t cycle_count_wrap = end_cycles_wrap - start_cycles_wrap;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
        printf("fc_per_ch_arm_fully_connected_wrapper_s8 output validation PASSED (wrapper)\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used_wrap);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count_wrap);
    } else {
        printf("fc_per_ch_arm_fully_connected_wrapper_s8 output validation FAILED (wrapper)\n\r");
    }
}

void fully_connected_mve_0_arm_fully_connected_s8(void)
{
    int8_t output[FULLY_CONNECTED_MVE_0_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = fully_connected_mve_0_biases;
    const int8_t *kernel_data = fully_connected_mve_0_weights;
    const int8_t *input_data = fully_connected_mve_0_input;
    const int8_t *output_ref = fully_connected_mve_0_output_ref;
    const int32_t output_ref_size = FULLY_CONNECTED_MVE_0_DST_SIZE;

    input_dims.n = FULLY_CONNECTED_MVE_0_INPUT_BATCHES;
    input_dims.w = FULLY_CONNECTED_MVE_0_INPUT_W;
    input_dims.h = FULLY_CONNECTED_MVE_0_INPUT_H;
    input_dims.c = FULLY_CONNECTED_MVE_0_IN_CH;
    filter_dims.n = FULLY_CONNECTED_MVE_0_ACCUMULATION_DEPTH;
    filter_dims.c = FULLY_CONNECTED_MVE_0_OUT_CH;
    output_dims.n = FULLY_CONNECTED_MVE_0_INPUT_BATCHES;
    output_dims.c = FULLY_CONNECTED_MVE_0_OUT_CH;

    fc_params.input_offset = FULLY_CONNECTED_MVE_0_INPUT_OFFSET;
    fc_params.filter_offset = 0;
    fc_params.output_offset = FULLY_CONNECTED_MVE_0_OUTPUT_OFFSET;
    fc_params.activation.min = FULLY_CONNECTED_MVE_0_OUT_ACTIVATION_MIN;
    fc_params.activation.max = FULLY_CONNECTED_MVE_0_OUT_ACTIVATION_MAX;

    quant_params.multiplier = FULLY_CONNECTED_MVE_0_OUTPUT_MULTIPLIER;
    quant_params.shift = FULLY_CONNECTED_MVE_0_OUTPUT_SHIFT;

    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

#if defined(ARM_MATH_MVEI)
    int32_t *buf = ctx.buf;
    arm_vector_sum_s8(buf,
                      filter_dims.n,
                      output_dims.c,
                      kernel_data,
                      fc_params.input_offset,
                      fc_params.filter_offset,
                      bias_data);
#endif

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles = read_cycle_counter();

    arm_fully_connected_s8(&ctx,
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
    if (validate(output, output_ref, output_ref_size)) {
        printf("fully_connected_mve_0_arm_fully_connected_s8 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("fully_connected_mve_0_arm_fully_connected_s8 output validation FAILED\n\r");
    }
}

