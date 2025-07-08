#include "main.h"

#include "TestData/transpose_conv_1/test_data.h"
#include "../TestData/transpose_conv_2/test_data.h"
#include "../TestData/transpose_conv_3/test_data.h"
#define REPEAT_NUM (2)

RAM_FUNC void transpose_conv_1_arm_transpose_conv_s8(void)
{
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[TRANSPOSE_CONV_1_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_1_biases;
    const int8_t *kernel_data = transpose_conv_1_weights;
    const int8_t *input_data = transpose_conv_1_input;
    const int8_t *output_ref = transpose_conv_1_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_1_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_1_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_1_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_1_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_1_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_1_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_1_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_1_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_1_IN_CH;
    output_dims.n = TRANSPOSE_CONV_1_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_1_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_1_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_1_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_1_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_1_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_1_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_1_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_1_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_1_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_1_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_1_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_1_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_1_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_1_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_1_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_1_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_1_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles = read_cycle_counter();

    arm_transpose_conv_wrapper_s8(&ctx,
								   &reverse_conv_ctx,
								   &transpose_conv_params,
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

    if (reverse_conv_ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
		printf("transpose_conv_1 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
	} else {
		printf("transpose_conv_1 output validation FAILED\n\r");
	}
}

RAM_FUNC void transpose_conv_2_arm_transpose_conv_s8(void)
{
    int8_t output[TRANSPOSE_CONV_2_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_2_biases;
    const int8_t *kernel_data = transpose_conv_2_weights;
    const int8_t *input_data = transpose_conv_2_input;
    const int8_t *output_ref = transpose_conv_2_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_2_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_2_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_2_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_2_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_2_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_2_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_2_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_2_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_2_IN_CH;
    output_dims.n = TRANSPOSE_CONV_2_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_2_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_2_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_2_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_2_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_2_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_2_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_2_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_2_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_2_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_2_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_2_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_2_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_2_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_2_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_2_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_2_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_2_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles = read_cycle_counter();

    arm_transpose_conv_wrapper_s8(&ctx,
                                   &reverse_conv_ctx,
                                   &transpose_conv_params,
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

    if (reverse_conv_ctx.buf)
    {
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    printf("\n\r");

    if (validate(output, output_ref, output_ref_size)) {
        printf("transpose_conv_2 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    } else {
        printf("transpose_conv_2 output validation FAILED\n\r");
    }
}

RAM_FUNC void transpose_conv_3_arm_transpose_conv_s8(void)
{
    int8_t output[TRANSPOSE_CONV_3_DST_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_context reverse_conv_ctx;
    cmsis_nn_transpose_conv_params transpose_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims bias_dims = {0};
    cmsis_nn_dims output_dims;

    const int32_t *bias_data = transpose_conv_3_biases;
    const int8_t *kernel_data = transpose_conv_3_weights;
    const int8_t *input_data = transpose_conv_3_input;
    const int8_t *output_ref = transpose_conv_3_output_ref;
    const int32_t output_ref_size = TRANSPOSE_CONV_3_DST_SIZE;

    input_dims.n = TRANSPOSE_CONV_3_INPUT_BATCHES;
    input_dims.w = TRANSPOSE_CONV_3_INPUT_W;
    input_dims.h = TRANSPOSE_CONV_3_INPUT_H;
    input_dims.c = TRANSPOSE_CONV_3_IN_CH;
    filter_dims.w = TRANSPOSE_CONV_3_FILTER_X;
    filter_dims.h = TRANSPOSE_CONV_3_FILTER_Y;
    filter_dims.n = TRANSPOSE_CONV_3_OUT_CH;
    filter_dims.c = TRANSPOSE_CONV_3_IN_CH;
    output_dims.n = TRANSPOSE_CONV_3_INPUT_BATCHES;
    output_dims.w = TRANSPOSE_CONV_3_OUTPUT_W;
    output_dims.h = TRANSPOSE_CONV_3_OUTPUT_H;
    output_dims.c = TRANSPOSE_CONV_3_OUT_CH;

    transpose_conv_params.padding.w = TRANSPOSE_CONV_3_PAD_X;
    transpose_conv_params.padding.h = TRANSPOSE_CONV_3_PAD_Y;
    transpose_conv_params.padding_offsets.w = TRANSPOSE_CONV_3_PAD_X_WITH_OFFSET;
    transpose_conv_params.padding_offsets.h = TRANSPOSE_CONV_3_PAD_Y_WITH_OFFSET;

    transpose_conv_params.stride.w = TRANSPOSE_CONV_3_STRIDE_X;
    transpose_conv_params.stride.h = TRANSPOSE_CONV_3_STRIDE_Y;
    transpose_conv_params.dilation.w = TRANSPOSE_CONV_3_DILATION_X;
    transpose_conv_params.dilation.h = TRANSPOSE_CONV_3_DILATION_Y;

    transpose_conv_params.input_offset = TRANSPOSE_CONV_3_INPUT_OFFSET;
    transpose_conv_params.output_offset = TRANSPOSE_CONV_3_OUTPUT_OFFSET;
    transpose_conv_params.activation.min = TRANSPOSE_CONV_3_OUT_ACTIVATION_MIN;
    transpose_conv_params.activation.max = TRANSPOSE_CONV_3_OUT_ACTIVATION_MAX;
    quant_params.multiplier = (int32_t *)transpose_conv_3_output_mult;
    quant_params.shift = (int32_t *)transpose_conv_3_output_shift;

    const int32_t buf_size =
        arm_transpose_conv_s8_get_buffer_size(&transpose_conv_params, &input_dims, &filter_dims, &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    const int32_t reverse_conv_buf_size =
        arm_transpose_conv_s8_get_reverse_conv_buffer_size(&transpose_conv_params, &input_dims, &filter_dims);
    reverse_conv_ctx.buf = malloc(reverse_conv_buf_size);
    reverse_conv_ctx.size = reverse_conv_buf_size;

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles = read_cycle_counter();

    arm_transpose_conv_wrapper_s8(&ctx,
                                   &reverse_conv_ctx,
                                   &transpose_conv_params,
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

    if (reverse_conv_ctx.buf)
    {
        memset(reverse_conv_ctx.buf, 0, reverse_conv_ctx.size);
        free(reverse_conv_ctx.buf);
    }

    if (ctx.buf)
    {
        memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
        printf("transpose_conv_3 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    } else {
        printf("transpose_conv_3 output validation FAILED\n\r");
    }
}

