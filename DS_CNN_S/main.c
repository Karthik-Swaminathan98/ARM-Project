#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_nnfunctions.h"
//#include "unity.h"
#include <string.h> // For memcpy
#include <stdlib.h>

#include "test_data.h"
#include "validate.h"

#define MAX_DIM_SIZE_BYTE_0 (CONV_2D_1_OUTPUT_W * CONV_2D_1_OUTPUT_H * CONV_2D_1_OUT_CH)
#define MAX_DIM_SIZE_BYTE_1 (DEPTHWISE_CONV_2D_2_OUTPUT_H * DEPTHWISE_CONV_2D_2_OUTPUT_W * DEPTHWISE_CONV_2D_2_OUT_CH)

#define MAX_SIZE_BYTES (MAX_DIM_SIZE_BYTE_0 > MAX_DIM_SIZE_BYTE_1 ? MAX_DIM_SIZE_BYTE_0 : MAX_DIM_SIZE_BYTE_1)

// Word aligned start addresses to prevent unalinged access.
#define MAX_NUM_WORDS_IN_OUT ((MAX_DIM_SIZE_BYTE_0 + MAX_DIM_SIZE_BYTE_1 + 3) / 4)
#define IN_OUT_BUFER_0_BYTE_OFFSET (0)
#define IN_OUT_BUFER_1_BYTE_OFFSET (MAX_SIZE_BYTES + MAX_SIZE_BYTES % 4)

static int32_t in_out_buf_main[MAX_NUM_WORDS_IN_OUT];

// Enable DWT cycle counter
static void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; // Enable DWT
    DWT->CYCCNT = 0;                                // Reset cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;            // Enable cycle counter
}

// Read DWT cycle counter
static uint32_t read_cycle_counter() {
    return DWT->CYCCNT;
}

// Fill stack with known pattern for usage measurement
static void fill_stack_pattern_to_sp() {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));

    extern uint32_t __StackLimit;
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) {
        *p++ = 0xAAAAAAAA;
    }
}

// Measure stack usage by identifying overwritten bytes
static uint32_t measure_stack_usage() {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));

    extern uint32_t __StackLimit;
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) {
        if (*p != 0xAAAAAAAA) {
            break;
        }
        p++;
    }
    return ((uint32_t)sp - (uint32_t)p); // Stack usage in bytes
}

/* Get size of additional buffers required by library/framework */
int ds_cnn_s_s8_get_buffer_size(void)
{
    /* Custom function based on knowledge that only select layers of DS_CNN_S network require additional buffers. */
    int max_buffer = 0;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_dims input_dims;
    cmsis_nn_dims filter_dims;
    cmsis_nn_dims output_dims;

    // Layer 0 - Conv
    conv_params.padding.h = CONV_2D_1_PAD_H;
    conv_params.padding.w = CONV_2D_1_PAD_W;
    conv_params.stride.h = CONV_2D_1_STRIDE_H;
    conv_params.stride.w = CONV_2D_1_STRIDE_W;
    conv_params.dilation.h = CONV_2D_1_DILATION_H;
    conv_params.dilation.w = CONV_2D_1_DILATION_W;

    input_dims.n = CONV_2D_1_INPUT_BATCHES;
    input_dims.h = CONV_2D_1_INPUT_H;
    input_dims.w = CONV_2D_1_INPUT_W;
    input_dims.c = CONV_2D_1_IN_CH;

    filter_dims.h = CONV_2D_1_FILTER_H;
    filter_dims.w = CONV_2D_1_FILTER_W;
    filter_dims.c = CONV_2D_1_IN_CH;

    output_dims.n = input_dims.n;
    output_dims.h = CONV_2D_1_OUTPUT_H;
    output_dims.w = CONV_2D_1_OUTPUT_W;
    output_dims.c = CONV_2D_1_OUT_CH;

    int32_t size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params, &input_dims, &filter_dims, &output_dims);

    max_buffer = size > max_buffer ? size : max_buffer;

    // Layer 0 - DW Conv
    cmsis_nn_dw_conv_params dw_conv_params;
    dw_conv_params.activation.min = DEPTHWISE_CONV_2D_2_OUT_ACTIVATION_MIN;
    dw_conv_params.activation.max = DEPTHWISE_CONV_2D_2_OUT_ACTIVATION_MAX;
    dw_conv_params.ch_mult = 1;
    dw_conv_params.dilation.h = DEPTHWISE_CONV_2D_2_DILATION_H;
    dw_conv_params.dilation.w = DEPTHWISE_CONV_2D_2_DILATION_W;
    dw_conv_params.input_offset = DEPTHWISE_CONV_2D_2_INPUT_OFFSET;
    dw_conv_params.output_offset = DEPTHWISE_CONV_2D_2_OUTPUT_OFFSET;
    dw_conv_params.padding.h = DEPTHWISE_CONV_2D_2_PAD_H;
    dw_conv_params.padding.w = DEPTHWISE_CONV_2D_2_PAD_W;
    dw_conv_params.stride.h = DEPTHWISE_CONV_2D_2_STRIDE_H;
    dw_conv_params.stride.w = DEPTHWISE_CONV_2D_2_STRIDE_W;

    filter_dims.h = DEPTHWISE_CONV_2D_2_FILTER_H;
    filter_dims.w = DEPTHWISE_CONV_2D_2_FILTER_W;

    input_dims.n = 1;
    input_dims.h = DEPTHWISE_CONV_2D_2_INPUT_H;
    input_dims.w = DEPTHWISE_CONV_2D_2_INPUT_W;
    input_dims.c = DEPTHWISE_CONV_2D_2_OUT_CH;

    output_dims.h = DEPTHWISE_CONV_2D_2_OUTPUT_H;
    output_dims.w = DEPTHWISE_CONV_2D_2_OUTPUT_W;
    output_dims.c = DEPTHWISE_CONV_2D_2_OUT_CH;

    size = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params, &input_dims, &filter_dims, &output_dims);

    max_buffer = size > max_buffer ? size : max_buffer;

    // Layer 12 - Fully connected
    size = FULLY_CONNECTED_12_OUTPUT_W * sizeof(int32_t);
    max_buffer = size > max_buffer ? size : max_buffer;

    return max_buffer;
}

int main(void)
{
    cy_rslt_t result;

    // Initialize the device and board peripherals
    result = cybsp_init();
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    // Enable global interrupts
    __enable_irq();

    // Initialize retarget-io to use the debug UART port
    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                                    CYBSP_DEBUG_UART_CTS, CYBSP_DEBUG_UART_RTS, CY_RETARGET_IO_BAUDRATE);

    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }
    printf("\r\n--- DS-CNN_S Inference on PSoC 6 M4 ---\r\n");
    uint32_t total_cycles = 0;
    uint32_t total_stack = 0;

    cmsis_nn_context ctx;
	//const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;

	ctx.size = ds_cnn_s_s8_get_buffer_size();
	ctx.buf = malloc(ctx.size);

	int8_t *in_out_buf_0 = (int8_t *)&in_out_buf_main[IN_OUT_BUFER_0_BYTE_OFFSET >> 2];
	int8_t *in_out_buf_1 = (int8_t *)&in_out_buf_main[IN_OUT_BUFER_1_BYTE_OFFSET >> 2];

	// Layer 0 - Implicit reshape
	// 1x490 is interpreted as 49x10

	// Layer 1 - Conv
	cmsis_nn_conv_params conv_params;
	cmsis_nn_per_channel_quant_params quant_params;
	cmsis_nn_dims in_out_dim_0;
	cmsis_nn_dims conv_filter_dims;
	cmsis_nn_dims dw_conv_filter_dims;
	cmsis_nn_dims in_out_dim_1;
	cmsis_nn_dims bias_dims;

	conv_params.padding.h = CONV_2D_1_PAD_H;
	conv_params.padding.w = CONV_2D_1_PAD_W;
	conv_params.stride.h = CONV_2D_1_STRIDE_H;
	conv_params.stride.w = CONV_2D_1_STRIDE_W;
	conv_params.dilation.h = CONV_2D_1_DILATION_H;
	conv_params.dilation.w = CONV_2D_1_DILATION_W;
	conv_params.input_offset = CONV_2D_1_INPUT_OFFSET;
	conv_params.output_offset = CONV_2D_1_OUTPUT_OFFSET;
	// Not repeated subsequently as it is the same for all in this specific case.
	conv_params.activation.min = -128;
	conv_params.activation.max = 127;

	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_1_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_1_conv_2d_output_shift;

	in_out_dim_0.n = CONV_2D_1_INPUT_BATCHES;
	in_out_dim_0.h = CONV_2D_1_INPUT_H;
	in_out_dim_0.w = CONV_2D_1_INPUT_W;
	in_out_dim_0.c = CONV_2D_1_IN_CH;

	conv_filter_dims.h = CONV_2D_1_FILTER_H;
	conv_filter_dims.w = CONV_2D_1_FILTER_W;
	conv_filter_dims.c = CONV_2D_1_IN_CH;

	in_out_dim_1.n = in_out_dim_0.n;
	in_out_dim_1.h = CONV_2D_1_OUTPUT_H;
	in_out_dim_1.w = CONV_2D_1_OUTPUT_W;
	in_out_dim_1.c = CONV_2D_1_OUT_CH;
	bias_dims.c = CONV_2D_1_OUT_CH;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    uint32_t start_cycles = read_cycle_counter();

	arm_convolve_wrapper_s8(&ctx,
							 &conv_params,
							 &quant_params,
							 &in_out_dim_0,
							 ds_cnn_s_input1,
							 &conv_filter_dims,
							 ds_cnn_s_layer_1_conv_2d_weights,
							 &bias_dims,
							 ds_cnn_s_layer_1_conv_2d_bias,
							 &in_out_dim_1,
							 in_out_buf_0);

    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 0: Layer 1 - Conv \n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	/***************************** Depthwise Separable Block 1 *************** */
	// Layer 1 - DW Conv
	// Common params for DW conv in subsequent layers
	cmsis_nn_dw_conv_params dw_conv_params;
	dw_conv_params.activation.min = DEPTHWISE_CONV_2D_2_OUT_ACTIVATION_MIN;
	dw_conv_params.activation.max = DEPTHWISE_CONV_2D_2_OUT_ACTIVATION_MAX;
	dw_conv_params.ch_mult = 1;
	dw_conv_params.dilation.h = DEPTHWISE_CONV_2D_2_DILATION_H;
	dw_conv_params.dilation.w = DEPTHWISE_CONV_2D_2_DILATION_W;
	dw_conv_params.padding.h = DEPTHWISE_CONV_2D_2_PAD_H;
	dw_conv_params.padding.w = DEPTHWISE_CONV_2D_2_PAD_W;
	dw_conv_params.stride.h = DEPTHWISE_CONV_2D_2_STRIDE_H;
	dw_conv_params.stride.w = DEPTHWISE_CONV_2D_2_STRIDE_W;

	// Layer specific params
	dw_conv_params.input_offset = DEPTHWISE_CONV_2D_2_INPUT_OFFSET;
	dw_conv_params.output_offset = DEPTHWISE_CONV_2D_2_OUTPUT_OFFSET;

	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_2_depthwise_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_2_depthwise_conv_2d_output_shift;

	dw_conv_filter_dims.h = DEPTHWISE_CONV_2D_2_FILTER_H;
	dw_conv_filter_dims.w = DEPTHWISE_CONV_2D_2_FILTER_W;

	in_out_dim_0.h = DEPTHWISE_CONV_2D_2_OUTPUT_H;
	in_out_dim_0.w = DEPTHWISE_CONV_2D_2_OUTPUT_W;
	in_out_dim_0.c = DEPTHWISE_CONV_2D_2_OUT_CH;

	// Same for all layers in DS block
	bias_dims.c = in_out_dim_0.c;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

	arm_depthwise_conv_wrapper_s8(&ctx,
								&dw_conv_params,
								&quant_params,
								&in_out_dim_1,
								in_out_buf_0,
								&dw_conv_filter_dims,
								ds_cnn_s_layer_2_depthwise_conv_2d_weights,
								&bias_dims,
								ds_cnn_s_layer_2_depthwise_conv_2d_bias,
								&in_out_dim_0,
								in_out_buf_1);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 1: Layer 1 - DW Conv\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	// Layer 2 - Conv

	// Common params for Conv in rest of DS blocks
	in_out_dim_1.h = in_out_dim_0.h;
	in_out_dim_1.w = in_out_dim_0.w;
	in_out_dim_1.c = in_out_dim_0.c;
	conv_filter_dims.h = CONV_2D_3_FILTER_H;
	conv_filter_dims.w = CONV_2D_3_FILTER_W;
	conv_filter_dims.c = CONV_2D_3_IN_CH;

	conv_params.padding.h = CONV_2D_3_PAD_H;
	conv_params.padding.w = CONV_2D_3_PAD_W;
	conv_params.stride.h = CONV_2D_3_STRIDE_H;
	conv_params.stride.w = CONV_2D_3_STRIDE_W;

	// Layer specific params
	conv_params.input_offset = CONV_2D_3_INPUT_OFFSET;
	conv_params.output_offset = CONV_2D_3_OUTPUT_OFFSET;

	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_3_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_3_conv_2d_output_shift;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

	arm_convolve_wrapper_s8(&ctx,
						  &conv_params,
						  &quant_params,
						  &in_out_dim_0,
						  in_out_buf_1,
						  &conv_filter_dims,
						  ds_cnn_s_layer_3_conv_2d_weights,
						  &bias_dims,
						  ds_cnn_s_layer_3_conv_2d_bias,
						  &in_out_dim_1,
						  in_out_buf_0);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 1: Layer 2 - Conv\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	/***************************** Depthwise Separable Block 2 *************** */
	// Layer specific
	dw_conv_params.input_offset = DEPTHWISE_CONV_2D_4_INPUT_OFFSET;
	dw_conv_params.output_offset = DEPTHWISE_CONV_2D_4_OUTPUT_OFFSET;

	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_4_depthwise_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_4_depthwise_conv_2d_output_shift;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

	arm_depthwise_conv_wrapper_s8(&ctx,
									&dw_conv_params,
									&quant_params,
									&in_out_dim_1,
									in_out_buf_0,
									&dw_conv_filter_dims,
									ds_cnn_s_layer_4_depthwise_conv_2d_weights,
									&bias_dims,
									ds_cnn_s_layer_4_depthwise_conv_2d_bias,
									&in_out_dim_0,
									in_out_buf_1);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 2: Layer 1 - DW Conv\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	// Layer specific params
	conv_params.input_offset = CONV_2D_5_INPUT_OFFSET;
	conv_params.output_offset = CONV_2D_5_OUTPUT_OFFSET;

	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_5_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_5_conv_2d_output_shift;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

	arm_convolve_wrapper_s8(&ctx,
							  &conv_params,
							  &quant_params,
							  &in_out_dim_0,
							  in_out_buf_1,
							  &conv_filter_dims,
							  ds_cnn_s_layer_5_conv_2d_weights,
							  &bias_dims,
							  ds_cnn_s_layer_5_conv_2d_bias,
							  &in_out_dim_1,
							  in_out_buf_0);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 2: Layer 2 - Conv\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	/***************************** Depthwise Separable Block 3 *************** */
	// Layer specific
	dw_conv_params.input_offset = DEPTHWISE_CONV_2D_6_INPUT_OFFSET;
	dw_conv_params.output_offset = DEPTHWISE_CONV_2D_6_OUTPUT_OFFSET;

	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_6_depthwise_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_6_depthwise_conv_2d_output_shift;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

	arm_depthwise_conv_wrapper_s8(&ctx,
								&dw_conv_params,
								&quant_params,
								&in_out_dim_1,
								in_out_buf_0,
								&dw_conv_filter_dims,
								ds_cnn_s_layer_6_depthwise_conv_2d_weights,
								&bias_dims,
								ds_cnn_s_layer_6_depthwise_conv_2d_bias,
								&in_out_dim_0,
								in_out_buf_1);
    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 3: Layer 1 - DW Conv\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

	// Layer specific params
	conv_params.input_offset = CONV_2D_7_INPUT_OFFSET;
	conv_params.output_offset = CONV_2D_7_OUTPUT_OFFSET;
	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_7_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_7_conv_2d_output_shift;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

	arm_convolve_wrapper_s8(&ctx,
							  &conv_params,
							  &quant_params,
							  &in_out_dim_0,
							  in_out_buf_1,
							  &conv_filter_dims,
							  ds_cnn_s_layer_7_conv_2d_weights,
							  &bias_dims,
							  ds_cnn_s_layer_7_conv_2d_bias,
							  &in_out_dim_1,
							  in_out_buf_0);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 3: Layer 2 - Conv\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	/***************************** Depthwise Separable Block 4 *************** */
	// Layer specific
	dw_conv_params.input_offset = DEPTHWISE_CONV_2D_8_INPUT_OFFSET;
	dw_conv_params.output_offset = DEPTHWISE_CONV_2D_8_OUTPUT_OFFSET;

	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_8_depthwise_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_8_depthwise_conv_2d_output_shift;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

	arm_depthwise_conv_wrapper_s8(&ctx,
									&dw_conv_params,
									&quant_params,
									&in_out_dim_1,
									in_out_buf_0,
									&dw_conv_filter_dims,
									ds_cnn_s_layer_8_depthwise_conv_2d_weights,
									&bias_dims,
									ds_cnn_s_layer_8_depthwise_conv_2d_bias,
									&in_out_dim_0,
									in_out_buf_1);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 4: Layer 1 - DW Conv\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	conv_params.input_offset = CONV_2D_9_INPUT_OFFSET;
	conv_params.output_offset = CONV_2D_9_OUTPUT_OFFSET;

	quant_params.multiplier = (int32_t *)ds_cnn_s_layer_9_conv_2d_output_mult;
	quant_params.shift = (int32_t *)ds_cnn_s_layer_9_conv_2d_output_shift;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();
	arm_convolve_wrapper_s8(&ctx,
							  &conv_params,
							  &quant_params,
							  &in_out_dim_0,
							  in_out_buf_1,
							  &conv_filter_dims,
							  ds_cnn_s_layer_9_conv_2d_weights,
							  &bias_dims,
							  ds_cnn_s_layer_9_conv_2d_bias,
							  &in_out_dim_1,
							  in_out_buf_0);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Block 4: Layer 2 - Conv\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	/***************************** Average Pool *************** */

	cmsis_nn_pool_params pool_params;
	pool_params.activation.max = AVERAGE_POOL_2D_10_OUT_ACTIVATION_MAX;
	pool_params.activation.min = AVERAGE_POOL_2D_10_OUT_ACTIVATION_MIN;
	pool_params.padding.h = AVERAGE_POOL_2D_10_PAD_H;
	pool_params.padding.w = AVERAGE_POOL_2D_10_PAD_W;
	pool_params.stride.h = AVERAGE_POOL_2D_10_STRIDE_H;
	pool_params.stride.w = AVERAGE_POOL_2D_10_STRIDE_W;

	conv_filter_dims.h = AVERAGE_POOL_2D_10_FILTER_H;
	conv_filter_dims.w = AVERAGE_POOL_2D_10_FILTER_W;

	in_out_dim_0.n = 1;
	in_out_dim_0.h = AVERAGE_POOL_2D_10_OUTPUT_H;
	in_out_dim_0.w = AVERAGE_POOL_2D_10_OUTPUT_W;
	in_out_dim_0.c = in_out_dim_1.c;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

    arm_avgpool_s8(&ctx, &pool_params, &in_out_dim_1, in_out_buf_0, &conv_filter_dims, &in_out_dim_0, in_out_buf_1);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Average Pool\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	/***************************** Fully Connected ****************/
	cmsis_nn_fc_params fc_params;
	fc_params.activation.max = FULLY_CONNECTED_12_OUT_ACTIVATION_MAX;
	fc_params.activation.min = FULLY_CONNECTED_12_OUT_ACTIVATION_MIN;
	fc_params.filter_offset = 0;
	fc_params.input_offset = FULLY_CONNECTED_12_INPUT_OFFSET;
	fc_params.output_offset = FULLY_CONNECTED_12_OUTPUT_OFFSET;

	cmsis_nn_per_tensor_quant_params per_tensor_quant_params;
	per_tensor_quant_params.multiplier = FULLY_CONNECTED_12_OUTPUT_MULTIPLIER;
	per_tensor_quant_params.shift = FULLY_CONNECTED_12_OUTPUT_SHIFT;

	in_out_dim_0.c = in_out_dim_0.c * in_out_dim_0.h * in_out_dim_0.w;
	in_out_dim_0.h = 1;
	in_out_dim_0.w = 1;

	conv_filter_dims.n = in_out_dim_0.c;
	conv_filter_dims.h = 1;
	conv_filter_dims.w = 1;
	conv_filter_dims.c = FULLY_CONNECTED_12_OUTPUT_W;

	in_out_dim_1.n = 1;
	in_out_dim_1.h = 1;
	in_out_dim_1.w = 1;
	in_out_dim_1.c = FULLY_CONNECTED_12_OUTPUT_W;

	bias_dims.c = in_out_dim_1.c;

#if defined(ARM_MATH_MVEI)
	arm_vector_sum_s8(ctx.buf,
					  conv_filter_dims.n,
					  in_out_dim_1.c,
					  ds_cnn_s_layer_12_fully_connected_weights,
					  fc_params.input_offset,
					  0,
					  ds_cnn_s_layer_12_fully_connected_bias);
#endif

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();

	arm_fully_connected_s8(&ctx,
							 &fc_params,
							 &per_tensor_quant_params,
							 &in_out_dim_0,
							 in_out_buf_1,
							 &conv_filter_dims,
							 ds_cnn_s_layer_12_fully_connected_weights,
							 &bias_dims,
							 ds_cnn_s_layer_12_fully_connected_bias,
							 &in_out_dim_1,
							 in_out_buf_0);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Fully Connected\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	/***************************** Softmax *************** */

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();
    start_cycles = read_cycle_counter();
	arm_softmax_s8(in_out_buf_0,
				   SOFTMAX_13_NUM_ROWS,
				   SOFTMAX_13_ROW_SIZE,
				   SOFTMAX_13_MULT,
				   SOFTMAX_13_SHIFT,
				   SOFTMAX_13_DIFF_MIN,
				   in_out_buf_0);

    end_cycles = read_cycle_counter();
    stack_used = measure_stack_usage();
    cycle_count = end_cycles - start_cycles;
    // Print results
    printf("\n\r");
    printf("Softmax\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    printf("-----DSCNN_SMALL benchmark complete-----\n\r");
    printf("Total Cycle Count: %lu\n\r", (unsigned long)total_cycles);
    printf("Total Stack Usage: %lu bytes\n\r", (unsigned long)total_stack);

    // Add to totals
    total_cycles += cycle_count;
    total_stack += stack_used;

	// After inference, buf0 holds the softmax output
	const char *labels[12] = {"yes","no","up","down","left","right","on","off","stop","go","unknown","silence"};
	int8_t *output = in_out_buf_0;
//	printf("Softmax output (12 classes):\r\n");
//	for (int i = 0; i < SOFTMAX_13_ROW_SIZE; ++i) {
//		printf("Label %d: %s	= %4d\r\n", i, labels[i], output[i]);
//	}

	// Validate against reference
	bool ok = validate(output,
					   ds_cnn_s_output_ref,
					   sizeof(ds_cnn_s_output_ref));
	printf("Validation %s\r\n", ok ? "PASSED" : "FAILED");

	free(ctx.buf);
	return 0;
}
