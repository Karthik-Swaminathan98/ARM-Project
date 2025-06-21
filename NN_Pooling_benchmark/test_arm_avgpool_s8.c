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
#include "TestData/avgpooling/test_data.h"
#include "TestData/avgpooling_1/test_data.h"
#include "TestData/avgpooling_2/test_data.h"

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

void avgpooling_arm_avgpool_s8(void)
{
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output[AVGPOOLING_OUTPUT_W * AVGPOOLING_OUTPUT_H * AVGPOOLING_BATCH_SIZE * AVGPOOLING_OUTPUT_C] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = avgpooling_input_tensor;

    input_dims.n = AVGPOOLING_BATCH_SIZE;
    input_dims.w = AVGPOOLING_INPUT_W;
    input_dims.h = AVGPOOLING_INPUT_H;
    input_dims.c = AVGPOOLING_INPUT_C;
    filter_dims.w = AVGPOOLING_FILTER_W;
    filter_dims.h = AVGPOOLING_FILTER_H;
    output_dims.w = AVGPOOLING_OUTPUT_W;
    output_dims.h = AVGPOOLING_OUTPUT_H;
    output_dims.c = AVGPOOLING_OUTPUT_C;

    pool_params.padding.w = AVGPOOLING_PADDING_W;
    pool_params.padding.h = AVGPOOLING_PADDING_H;
    pool_params.stride.w = AVGPOOLING_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_OUTPUT_W, AVGPOOLING_INPUT_C);
    ctx.buf = malloc(ctx.size);

	enable_cycle_counter();

	// Fill stack with a known pattern
	fill_stack_pattern_to_sp();

    // Measure cycles
    uint32_t start_cycles_s8 = read_cycle_counter();
    arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data, &filter_dims, &output_dims, output);
    // Measure cycles
    uint32_t end_cycles_s8 = read_cycle_counter();

    // Measure stack usage
    uint32_t stack_used_s8 = measure_stack_usage();

    // Calculate cycle count
    uint32_t cycle_count_s8 = end_cycles_s8 - start_cycles_s8;
    if (ctx.buf)
    {
        // The caller is responsible to clear the scratch buffers for security reasons if applicable.
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    printf("\n\r");
	if (validate(output,
            avgpooling_output,
            AVGPOOLING_OUTPUT_W * AVGPOOLING_OUTPUT_H * AVGPOOLING_BATCH_SIZE * AVGPOOLING_OUTPUT_C)) {
		printf("arm_avgpool_s8 output validation PASSED\n\r");
		printf("Stack Used for arm_avgpool_s8: %lu bytes\n\r", (unsigned long)stack_used_s8);
		printf("Cycle Count for arm_avgpool_s8: %lu\n\r", (unsigned long)cycle_count_s8);
	} else {
		printf("arm_avgpool_s8 output validation FAILED\n\r");
	}
}

void avgpooling_1_arm_avgpool_s8(void)
{
    int8_t output[AVGPOOLING_1_OUTPUT_W * AVGPOOLING_1_OUTPUT_H * AVGPOOLING_1_BATCH_SIZE * AVGPOOLING_1_OUTPUT_C] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = avgpooling_1_input_tensor;

    input_dims.n = AVGPOOLING_1_BATCH_SIZE;
    input_dims.w = AVGPOOLING_1_INPUT_W;
    input_dims.h = AVGPOOLING_1_INPUT_H;
    input_dims.c = AVGPOOLING_1_INPUT_C;
    filter_dims.w = AVGPOOLING_1_FILTER_W;
    filter_dims.h = AVGPOOLING_1_FILTER_H;
    output_dims.w = AVGPOOLING_1_OUTPUT_W;
    output_dims.h = AVGPOOLING_1_OUTPUT_H;
    output_dims.c = AVGPOOLING_1_OUTPUT_C;

    pool_params.padding.w = AVGPOOLING_1_PADDING_W;
    pool_params.padding.h = AVGPOOLING_1_PADDING_H;
    pool_params.stride.w = AVGPOOLING_1_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_1_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_1_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_1_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_1_OUTPUT_W, AVGPOOLING_1_INPUT_C);
    ctx.buf = malloc(ctx.size);

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
    arm_cmsis_nn_status result = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data,
                                                &filter_dims, &output_dims, output);
    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;

    if (ctx.buf) {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (result == ARM_CMSIS_NN_SUCCESS && validate(output, avgpooling_1_output,
        AVGPOOLING_1_OUTPUT_W * AVGPOOLING_1_OUTPUT_H * AVGPOOLING_1_BATCH_SIZE * AVGPOOLING_1_OUTPUT_C)) {
        printf("avgpooling_1_arm_avgpool_s8 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("avgpooling_1_arm_avgpool_s8 output validation FAILED\n\r");
    }
}

void avgpooling_2_arm_avgpool_s8(void)
{
    int8_t output[AVGPOOLING_2_OUTPUT_W * AVGPOOLING_2_OUTPUT_H * AVGPOOLING_2_BATCH_SIZE * AVGPOOLING_2_OUTPUT_C] = {0};

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    const int8_t *input_data = avgpooling_2_input_tensor;

    input_dims.n = AVGPOOLING_2_BATCH_SIZE;
    input_dims.w = AVGPOOLING_2_INPUT_W;
    input_dims.h = AVGPOOLING_2_INPUT_H;
    input_dims.c = AVGPOOLING_2_INPUT_C;
    filter_dims.w = AVGPOOLING_2_FILTER_W;
    filter_dims.h = AVGPOOLING_2_FILTER_H;
    output_dims.w = AVGPOOLING_2_OUTPUT_W;
    output_dims.h = AVGPOOLING_2_OUTPUT_H;
    output_dims.c = AVGPOOLING_2_OUTPUT_C;

    pool_params.padding.w = AVGPOOLING_2_PADDING_W;
    pool_params.padding.h = AVGPOOLING_2_PADDING_H;
    pool_params.stride.w = AVGPOOLING_2_STRIDE_W;
    pool_params.stride.h = AVGPOOLING_2_STRIDE_H;

    pool_params.activation.min = AVGPOOLING_2_ACTIVATION_MIN;
    pool_params.activation.max = AVGPOOLING_2_ACTIVATION_MAX;

    ctx.size = arm_avgpool_s8_get_buffer_size(AVGPOOLING_2_OUTPUT_W, AVGPOOLING_2_INPUT_C);
    ctx.buf = malloc(ctx.size);

    enable_cycle_counter();
    fill_stack_pattern_to_sp();

    uint32_t start_cycles = read_cycle_counter();
    arm_cmsis_nn_status result = arm_avgpool_s8(&ctx, &pool_params, &input_dims, input_data,
                                                &filter_dims, &output_dims, output);
    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;

    if (ctx.buf) {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }
    printf("\n\r");
    if (result == ARM_CMSIS_NN_SUCCESS && validate(output, avgpooling_2_output,
        AVGPOOLING_2_OUTPUT_W * AVGPOOLING_2_OUTPUT_H * AVGPOOLING_2_BATCH_SIZE * AVGPOOLING_2_OUTPUT_C)) {
        printf("avgpooling_2_arm_avgpool_s8 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    } else {
        printf("avgpooling_2_arm_avgpool_s8 output validation FAILED\n\r");
    }
}
