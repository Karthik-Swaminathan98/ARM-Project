#include "main.h"
#include "TestData/avgpooling_int16/test_data.h"
#include "TestData/avgpooling_int16_1/test_data.h"
#include "TestData/avgpooling_int16_2/test_data.h"

RAM_FUNC void avgpooling_int16_arm_avgpool_s16(void)
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

    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    uint32_t start_cycles = read_cycle_counter();

    arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

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
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (validate_s16(output,
            avgpooling_int16_output,
            AVGPOOLING_INT16_OUTPUT_C * AVGPOOLING_INT16_OUTPUT_W * AVGPOOLING_INT16_OUTPUT_H *
                AVGPOOLING_INT16_BATCH_SIZE)) {
		printf("arm_avgpool_s16 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
	} else {
		printf("arm_avgpool_s16 output validation FAILED\n\r");
	}
}

RAM_FUNC void avgpooling_int16_1_arm_avgpool_s16(void)
{
    int16_t output[AVGPOOLING_INT16_1_OUTPUT_C * AVGPOOLING_INT16_1_OUTPUT_W * AVGPOOLING_INT16_1_OUTPUT_H *
                   AVGPOOLING_INT16_1_BATCH_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_data = avgpooling_int16_1_input_tensor;

    input_dims.n = AVGPOOLING_INT16_1_BATCH_SIZE;
    input_dims.w = AVGPOOLING_INT16_1_INPUT_W;
    input_dims.h = AVGPOOLING_INT16_1_INPUT_H;
    input_dims.c = AVGPOOLING_INT16_1_INPUT_C;
    filter_dims.w = AVGPOOLING_INT16_1_FILTER_W;
    filter_dims.h = AVGPOOLING_INT16_1_FILTER_H;
    output_dims.w = AVGPOOLING_INT16_1_OUTPUT_W;
    output_dims.h = AVGPOOLING_INT16_1_OUTPUT_H;
    output_dims.c = AVGPOOLING_INT16_1_INPUT_C;

    pool_params.padding.w = AVGPOOLING_INT16_1_PADDING_W;
    pool_params.padding.h = AVGPOOLING_INT16_1_PADDING_H;
    pool_params.stride.w = AVGPOOLING_INT16_1_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_INT16_1_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_INT16_1_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_INT16_1_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_1_OUTPUT_W, AVGPOOLING_INT16_1_INPUT_C);
    ctx.buf = malloc(ctx.size);

    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    uint32_t start_cycles = read_cycle_counter();

    arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (validate_s16(output,
                     avgpooling_int16_1_output,
                     AVGPOOLING_INT16_1_OUTPUT_C * AVGPOOLING_INT16_1_OUTPUT_W *
                         AVGPOOLING_INT16_1_OUTPUT_H * AVGPOOLING_INT16_1_BATCH_SIZE))
    {
        printf("arm_avgpool_s16 (1) output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    }
    else
    {
        printf("arm_avgpool_s16 (1) output validation FAILED\n\r");
    }
}

RAM_FUNC void avgpooling_int16_2_arm_avgpool_s16(void)
{
    int16_t output[AVGPOOLING_INT16_2_OUTPUT_C * AVGPOOLING_INT16_2_OUTPUT_W * AVGPOOLING_INT16_2_OUTPUT_H *
                   AVGPOOLING_INT16_2_BATCH_SIZE] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int16_t *input_data = avgpooling_int16_2_input_tensor;

    input_dims.n = AVGPOOLING_INT16_2_BATCH_SIZE;
    input_dims.w = AVGPOOLING_INT16_2_INPUT_W;
    input_dims.h = AVGPOOLING_INT16_2_INPUT_H;
    input_dims.c = AVGPOOLING_INT16_2_INPUT_C;
    filter_dims.w = AVGPOOLING_INT16_2_FILTER_W;
    filter_dims.h = AVGPOOLING_INT16_2_FILTER_H;
    output_dims.w = AVGPOOLING_INT16_2_OUTPUT_W;
    output_dims.h = AVGPOOLING_INT16_2_OUTPUT_H;
    output_dims.c = AVGPOOLING_INT16_2_INPUT_C;

    pool_params.padding.w = AVGPOOLING_INT16_2_PADDING_W;
    pool_params.padding.h = AVGPOOLING_INT16_2_PADDING_H;
    pool_params.stride.w = AVGPOOLING_INT16_2_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_INT16_2_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_INT16_2_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_INT16_2_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s16_get_buffer_size(AVGPOOLING_INT16_2_OUTPUT_W, AVGPOOLING_INT16_2_INPUT_C);
    ctx.buf = malloc(ctx.size);

    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    uint32_t start_cycles = read_cycle_counter();

    arm_avgpool_s16(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);

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
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (validate_s16(output,
                     avgpooling_int16_2_output,
                     AVGPOOLING_INT16_2_OUTPUT_C * AVGPOOLING_INT16_2_OUTPUT_W *
                         AVGPOOLING_INT16_2_OUTPUT_H * AVGPOOLING_INT16_2_BATCH_SIZE))
    {
        printf("arm_avgpool_s16 (2) output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    }
    else
    {
        printf("arm_avgpool_s16 (2) output validation FAILED\n\r");
    }
}


