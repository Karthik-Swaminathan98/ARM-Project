#include "main.h"
#include "../TestData/basic/test_data.h"
#include "../TestData/conv_2x2_dilation/test_data.h"
#include "../TestData/conv_3x3_dilation_5x5_input/test_data.h"

RAM_FUNC void basic_arm_convolve_s8(void)
{
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

    // Set input, filter, and output dimensions
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

    // Set convolution parameters
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

    // Get required buffer size
    int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    uint32_t start_cycles = read_cycle_counter();

    arm_convolve_wrapper_s8(&ctx,
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

    uint32_t end_cycles = read_cycle_counter();
    uint32_t cycle_count = end_cycles - start_cycles;
    uint32_t instr_est = cycle_count
                       - DWT->CPICNT
                       - DWT->EXCCNT
                       - DWT->SLEEPCNT
                       - DWT->LSUCNT
                       + DWT->FOLDCNT;
    uint32_t stack_used = measure_stack_usage();
    float time_sec = (float)cycle_count / clkFastfreq;
    float time_us = time_sec * 1e6f;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);  // Security wipe
        free(ctx.buf);
    }

    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
        printf("basic_arm_convolve_s8 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    } else {
        printf("basic_arm_convolve_s8 output validation FAILED\n\r");
    }
}

RAM_FUNC void conv_2x2_dilation_arm_convolve_s8(void)
{
    int8_t output[CONV_2X2_DILATION_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_2x2_dilation_biases;
    const int8_t *kernel_data = conv_2x2_dilation_weights;
    const int8_t *input_data = conv_2x2_dilation_input;
    const int8_t *output_ref = conv_2x2_dilation_output_ref;
    const int32_t output_ref_size = CONV_2X2_DILATION_DST_SIZE;

    input_dims.n = CONV_2X2_DILATION_INPUT_BATCHES;
    input_dims.w = CONV_2X2_DILATION_INPUT_W;
    input_dims.h = CONV_2X2_DILATION_INPUT_H;
    input_dims.c = CONV_2X2_DILATION_IN_CH;
    filter_dims.w = CONV_2X2_DILATION_FILTER_X;
    filter_dims.h = CONV_2X2_DILATION_FILTER_Y;
    filter_dims.c = CONV_2X2_DILATION_IN_CH;
    output_dims.w = CONV_2X2_DILATION_OUTPUT_W;
    output_dims.h = CONV_2X2_DILATION_OUTPUT_H;
    output_dims.c = CONV_2X2_DILATION_OUT_CH;

    conv_params.padding.w = CONV_2X2_DILATION_PAD_X;
    conv_params.padding.h = CONV_2X2_DILATION_PAD_Y;
    conv_params.stride.w = CONV_2X2_DILATION_STRIDE_X;
    conv_params.stride.h = CONV_2X2_DILATION_STRIDE_Y;
    conv_params.dilation.w = CONV_2X2_DILATION_DILATION_X;
    conv_params.dilation.h = CONV_2X2_DILATION_DILATION_Y;
    conv_params.input_offset = CONV_2X2_DILATION_INPUT_OFFSET;
    conv_params.output_offset = CONV_2X2_DILATION_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_2X2_DILATION_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_2X2_DILATION_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t *)conv_2x2_dilation_output_mult;
    quant_params.shift = (int32_t *)conv_2x2_dilation_output_shift;

    int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    uint32_t start_cycles = read_cycle_counter();

    arm_convolve_wrapper_s8(&ctx,
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

    uint32_t end_cycles = read_cycle_counter();
    uint32_t cycle_count = end_cycles - start_cycles;
    uint32_t instr_est = cycle_count
                       - DWT->CPICNT
                       - DWT->EXCCNT
                       - DWT->SLEEPCNT
                       - DWT->LSUCNT
                       + DWT->FOLDCNT;
    uint32_t stack_used = measure_stack_usage();
    float time_sec = (float)cycle_count / clkFastfreq;
    float time_us = time_sec * 1e6f;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
        printf("conv_2x2_dilation_arm_convolve_s8 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    } else {
        printf("conv_2x2_dilation_arm_convolve_s8 output validation FAILED\n\r");
    }
}

RAM_FUNC void conv_3x3_dilation_5x5_input_arm_convolve_s8(void)
{
    int8_t output[CONV_3X3_DILATION_5X5_INPUT_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims;
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = conv_3x3_dilation_5x5_input_biases;
    const int8_t *kernel_data = conv_3x3_dilation_5x5_input_weights;
    const int8_t *input_data = conv_3x3_dilation_5x5_input_input;
    const int8_t *output_ref = conv_3x3_dilation_5x5_input_output_ref;
    const int32_t output_ref_size = CONV_3X3_DILATION_5X5_INPUT_DST_SIZE;

    input_dims.n = CONV_3X3_DILATION_5X5_INPUT_INPUT_BATCHES;
    input_dims.w = CONV_3X3_DILATION_5X5_INPUT_INPUT_W;
    input_dims.h = CONV_3X3_DILATION_5X5_INPUT_INPUT_H;
    input_dims.c = CONV_3X3_DILATION_5X5_INPUT_IN_CH;
    filter_dims.w = CONV_3X3_DILATION_5X5_INPUT_FILTER_X;
    filter_dims.h = CONV_3X3_DILATION_5X5_INPUT_FILTER_Y;
    filter_dims.c = CONV_3X3_DILATION_5X5_INPUT_IN_CH;
    output_dims.w = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_W;
    output_dims.h = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_H;
    output_dims.c = CONV_3X3_DILATION_5X5_INPUT_OUT_CH;

    conv_params.padding.w = CONV_3X3_DILATION_5X5_INPUT_PAD_X;
    conv_params.padding.h = CONV_3X3_DILATION_5X5_INPUT_PAD_Y;
    conv_params.stride.w = CONV_3X3_DILATION_5X5_INPUT_STRIDE_X;
    conv_params.stride.h = CONV_3X3_DILATION_5X5_INPUT_STRIDE_Y;
    conv_params.dilation.w = CONV_3X3_DILATION_5X5_INPUT_DILATION_X;
    conv_params.dilation.h = CONV_3X3_DILATION_5X5_INPUT_DILATION_Y;
    conv_params.input_offset = CONV_3X3_DILATION_5X5_INPUT_INPUT_OFFSET;
    conv_params.output_offset = CONV_3X3_DILATION_5X5_INPUT_OUTPUT_OFFSET;
    conv_params.activation.min = CONV_3X3_DILATION_5X5_INPUT_OUT_ACTIVATION_MIN;
    conv_params.activation.max = CONV_3X3_DILATION_5X5_INPUT_OUT_ACTIVATION_MAX;

    quant_params.multiplier = (int32_t *)conv_3x3_dilation_5x5_input_output_mult;
    quant_params.shift = (int32_t *)conv_3x3_dilation_5x5_input_output_shift;

    int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = 0;

    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    uint32_t start_cycles = read_cycle_counter();

    arm_convolve_wrapper_s8(&ctx,
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

    uint32_t end_cycles = read_cycle_counter();
    uint32_t cycle_count = end_cycles - start_cycles;
    uint32_t instr_est = cycle_count
                       - DWT->CPICNT
                       - DWT->EXCCNT
                       - DWT->SLEEPCNT
                       - DWT->LSUCNT
                       + DWT->FOLDCNT;
    uint32_t stack_used = measure_stack_usage();
    float time_sec = (float)cycle_count / clkFastfreq;
    float time_us = time_sec * 1e6f;

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
        printf("conv_3x3_dilation_5x5_input_arm_convolve_s8 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    } else {
        printf("conv_3x3_dilation_5x5_input_arm_convolve_s8 output validation FAILED\n\r");
    }
}

