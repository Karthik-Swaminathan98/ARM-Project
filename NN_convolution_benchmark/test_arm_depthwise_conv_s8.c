#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../TestData/depthwise_2/test_data.h"
#include "../TestData/depthwise_dilation/test_data.h"
#include "../TestData/depthwise_mult_batches/test_data.h"
#include "validate.h"

#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

static inline const int32_t *get_bias_address(const int32_t *bias, int32_t size)
{
    const int32_t *return_bias = NULL;
    for (int i = 0; i < size; i++)
    {
        if (bias[i] != 0)
        {
            return_bias = bias;
            break;
        }
    }
    return return_bias;
}

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

void depthwise_2_arm_depthwise_conv_s8(void)
{
    int8_t output[DEPTHWISE_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_2_DST_SIZE;
    const int32_t *bias_data = get_bias_address(depthwise_2_biases, DEPTHWISE_2_OUT_CH);
    const int8_t *kernel_data = depthwise_2_weights;
    const int8_t *input_data = depthwise_2_input;
    const int8_t *output_ref = depthwise_2_output_ref;

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

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
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
        printf("depthwise_2_arm_depthwise_conv_s8 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("depthwise_2_arm_depthwise_conv_s8 output validation FAILED\n\r");
    }
}

void depthwise_mult_batches_arm_depthwise_conv_s8(void)
{
    int8_t output[DEPTHWISE_MULT_BATCHES_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_MULT_BATCHES_DST_SIZE;
    const int32_t *bias_data = get_bias_address(depthwise_mult_batches_biases, DEPTHWISE_MULT_BATCHES_OUT_CH);
    const int8_t *kernel_data = depthwise_mult_batches_weights;
    const int8_t *input_data = depthwise_mult_batches_input;
    const int8_t *output_ref = depthwise_mult_batches_output_ref;

    input_dims.n = DEPTHWISE_MULT_BATCHES_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_MULT_BATCHES_INPUT_W;
    input_dims.h = DEPTHWISE_MULT_BATCHES_INPUT_H;
    input_dims.c = DEPTHWISE_MULT_BATCHES_IN_CH;
    filter_dims.w = DEPTHWISE_MULT_BATCHES_FILTER_X;
    filter_dims.h = DEPTHWISE_MULT_BATCHES_FILTER_Y;
    output_dims.w = DEPTHWISE_MULT_BATCHES_OUTPUT_W;
    output_dims.h = DEPTHWISE_MULT_BATCHES_OUTPUT_H;
    output_dims.c = DEPTHWISE_MULT_BATCHES_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_MULT_BATCHES_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_MULT_BATCHES_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_MULT_BATCHES_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_MULT_BATCHES_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_MULT_BATCHES_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_MULT_BATCHES_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_MULT_BATCHES_CH_MULT;
    dw_conv_params.input_offset = DEPTHWISE_MULT_BATCHES_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_MULT_BATCHES_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_MULT_BATCHES_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_MULT_BATCHES_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_mult_batches_output_mult;
    quant_params.shift = (int32_t *)depthwise_mult_batches_output_shift;

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
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
        printf("depthwise_mult_batches_arm_depthwise_conv_s8 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("depthwise_mult_batches_arm_depthwise_conv_s8 output validation FAILED\n\r");
    }
}

void depthwise_dilation_arm_depthwise_conv_s8(void)
{
    int8_t output[DEPTHWISE_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {};
    cmsis_nn_dims output_dims;

    const int32_t output_ref_size = DEPTHWISE_DILATION_DST_SIZE;
    const int32_t *bias_data = get_bias_address(depthwise_dilation_biases, DEPTHWISE_DILATION_OUT_CH);
    const int8_t *kernel_data = depthwise_dilation_weights;
    const int8_t *input_data = depthwise_dilation_input;
    const int8_t *output_ref = depthwise_dilation_output_ref;

    input_dims.n = DEPTHWISE_DILATION_INPUT_BATCHES;
    input_dims.w = DEPTHWISE_DILATION_INPUT_W;
    input_dims.h = DEPTHWISE_DILATION_INPUT_H;
    input_dims.c = DEPTHWISE_DILATION_IN_CH;
    filter_dims.w = DEPTHWISE_DILATION_FILTER_X;
    filter_dims.h = DEPTHWISE_DILATION_FILTER_Y;
    output_dims.w = DEPTHWISE_DILATION_OUTPUT_W;
    output_dims.h = DEPTHWISE_DILATION_OUTPUT_H;
    output_dims.c = DEPTHWISE_DILATION_OUT_CH;

    dw_conv_params.padding.w = DEPTHWISE_DILATION_PAD_X;
    dw_conv_params.padding.h = DEPTHWISE_DILATION_PAD_Y;
    dw_conv_params.stride.w = DEPTHWISE_DILATION_STRIDE_X;
    dw_conv_params.stride.h = DEPTHWISE_DILATION_STRIDE_Y;
    dw_conv_params.dilation.w = DEPTHWISE_DILATION_DILATION_X;
    dw_conv_params.dilation.h = DEPTHWISE_DILATION_DILATION_Y;

    dw_conv_params.ch_mult = DEPTHWISE_DILATION_CH_MULT;
    dw_conv_params.input_offset = DEPTHWISE_DILATION_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_DILATION_OUTPUT_OFFSET;
    dw_conv_params.activation.min = DEPTHWISE_DILATION_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_DILATION_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)depthwise_dilation_output_mult;
    quant_params.shift = (int32_t *)depthwise_dilation_output_shift;

    const int32_t buf_size =
        arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
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
        printf("depthwise_dilation_arm_depthwise_conv_s8 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("depthwise_dilation_arm_depthwise_conv_s8 output validation FAILED\n\r");
    }
}

