#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "../TestData/int16xint8/test_data.h"
#include "../TestData/int16xint8_dilation_1/test_data.h"
#include "../TestData/int16xint8xint32_1/test_data.h"
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

void basic_arm_convolve_s16(void)
{
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

    int buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();

    arm_cmsis_nn_status result = arm_convolve_wrapper_s16(&ctx,
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

    uint32_t end_cycles = read_cycle_counter();
    uint32_t cycle_count = end_cycles - start_cycles;
    uint32_t stack_used = measure_stack_usage();

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    printf("\n\r");
    if (result == ARM_CMSIS_NN_SUCCESS && validate_s16(output, output_ref, output_ref_size))
    {
        printf("basic_arm_convolve_s16 output validation PASSED\n\r");
        printf("Stack Used for basic_arm_convolve_s16: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count for basic_arm_convolve_s16: %lu\n\r", (unsigned long)cycle_count);
    }
    else
    {
        printf("basic_arm_convolve_s16 output validation FAILED\n\r");
    }
}


void int16xint8_dilation_1_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8_DILATION_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int64_t *int64_bias_data = int16xint8_dilation_1_biases;
    const cmsis_nn_bias_data bias_data = {int64_bias_data, false};
    const int8_t *kernel_data = int16xint8_dilation_1_weights;
    const int16_t *input_data = int16xint8_dilation_1_input;
    const int16_t *output_ref = int16xint8_dilation_1_output_ref;
    const int32_t output_ref_size = INT16XINT8_DILATION_1_DST_SIZE;

    input_dims.n = INT16XINT8_DILATION_1_INPUT_BATCHES;
    input_dims.w = INT16XINT8_DILATION_1_INPUT_W;
    input_dims.h = INT16XINT8_DILATION_1_INPUT_H;
    input_dims.c = INT16XINT8_DILATION_1_IN_CH;
    filter_dims.w = INT16XINT8_DILATION_1_FILTER_X;
    filter_dims.h = INT16XINT8_DILATION_1_FILTER_Y;
    output_dims.w = INT16XINT8_DILATION_1_OUTPUT_W;
    output_dims.h = INT16XINT8_DILATION_1_OUTPUT_H;
    output_dims.c = INT16XINT8_DILATION_1_OUT_CH;

    conv_params.padding.w = INT16XINT8_DILATION_1_PAD_X;
    conv_params.padding.h = INT16XINT8_DILATION_1_PAD_Y;
    conv_params.stride.w = INT16XINT8_DILATION_1_STRIDE_X;
    conv_params.stride.h = INT16XINT8_DILATION_1_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8_DILATION_1_DILATION_X;
    conv_params.dilation.h = INT16XINT8_DILATION_1_DILATION_Y;

    conv_params.input_offset = INT16XINT8_DILATION_1_INPUT_OFFSET;
    conv_params.output_offset = INT16XINT8_DILATION_1_OUTPUT_OFFSET;
    conv_params.activation.min = INT16XINT8_DILATION_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8_DILATION_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8_dilation_1_output_mult;
    quant_params.shift = (int32_t *)int16xint8_dilation_1_output_shift;

    int buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
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
        printf("int16xint8_dilation_1 arm_convolve_wrapper_s16 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("int16xint8_dilation_1 arm_convolve_wrapper_s16 output validation FAILED\n\r");
    }
}

void int16xint8xint32_1_arm_convolve_s16(void)
{
    int16_t output[INT16XINT8XINT32_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *int32_bias_data = int16xint8xint32_1_biases;
    const cmsis_nn_bias_data bias_data = {int32_bias_data, true};
    const int8_t *kernel_data = int16xint8xint32_1_weights;
    const int16_t *input_data = int16xint8xint32_1_input;
    const int16_t *output_ref = int16xint8xint32_1_output_ref;
    const int32_t output_ref_size = INT16XINT8XINT32_1_DST_SIZE;

    input_dims.n = INT16XINT8XINT32_1_INPUT_BATCHES;
    input_dims.w = INT16XINT8XINT32_1_INPUT_W;
    input_dims.h = INT16XINT8XINT32_1_INPUT_H;
    input_dims.c = INT16XINT8XINT32_1_IN_CH;
    filter_dims.w = INT16XINT8XINT32_1_FILTER_X;
    filter_dims.h = INT16XINT8XINT32_1_FILTER_Y;
    output_dims.w = INT16XINT8XINT32_1_OUTPUT_W;
    output_dims.h = INT16XINT8XINT32_1_OUTPUT_H;
    output_dims.c = INT16XINT8XINT32_1_OUT_CH;

    conv_params.padding.w = INT16XINT8XINT32_1_PAD_X;
    conv_params.padding.h = INT16XINT8XINT32_1_PAD_Y;
    conv_params.stride.w = INT16XINT8XINT32_1_STRIDE_X;
    conv_params.stride.h = INT16XINT8XINT32_1_STRIDE_Y;
    conv_params.dilation.w = INT16XINT8XINT32_1_DILATION_X;
    conv_params.dilation.h = INT16XINT8XINT32_1_DILATION_Y;

    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = INT16XINT8XINT32_1_OUT_ACTIVATION_MIN;
    conv_params.activation.max = INT16XINT8XINT32_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)int16xint8xint32_1_output_mult;
    quant_params.shift = (int32_t *)int16xint8xint32_1_output_shift;

    int buf_size = arm_convolve_wrapper_s16_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
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
        printf("int16xint8xint32_1 arm_convolve_wrapper_s16 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("int16xint8xint32_1 arm_convolve_wrapper_s16 output validation FAILED\n\r");
    }
}

