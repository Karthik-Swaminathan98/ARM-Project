#include <stdlib.h>
#include <arm_nnfunctions.h>
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <string.h>

// Function to generate random data for the buffers
void fill_dummy_data(int8_t *input, int8_t *weights, int32_t *biases, int32_t *output_mult, int32_t *output_shift,
                     size_t input_size, size_t weight_size, size_t bias_size) {
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = (rand() % 255) - 128; // random int8_t
    }
    for (size_t i = 0; i < weight_size; ++i) {
        weights[i] = (rand() % 255) - 128; // random int8_t
    }
    for (size_t i = 0; i < bias_size; ++i) {
        biases[i] = (rand() % 32768) - 16384; // random int32_t
        output_mult[i] = 0x40000000;  // Q31 format: multiplier = 1.0
        output_shift[i] = 0;          // No shift
    }
}

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

void layer1_arm_depthwise_conv_s8(uint32_t *total_cycles, uint32_t *total_stack) {
    // Define tensor and layer parameters for Layer 1
    const int input_h = 58, input_w = 13, input_c = 1;
    const int kernel_h = 10, kernel_w = 4, ch_mult = 64;
    const int output_h = 25, output_w = 5, output_c = 64;

    const size_t input_size = input_h * input_w * input_c;
    const size_t weight_size = kernel_h * kernel_w * input_c * ch_mult;
    const size_t bias_size = output_c;
    const size_t output_size = output_h * output_w * output_c;

    // Allocate memory for inputs, weights, biases, and output
    int8_t *depthwise_2_input = malloc(input_size);
    int8_t *depthwise_2_weights = malloc(weight_size);
    int32_t *depthwise_2_biases = malloc(bias_size * sizeof(int32_t));
    int32_t *depthwise_2_output_mult = malloc(bias_size * sizeof(int32_t));
    int32_t *depthwise_2_output_shift = malloc(bias_size * sizeof(int32_t));
    int8_t *output = malloc(output_size);

    // Fill buffers with dummy data
    fill_dummy_data(depthwise_2_input, depthwise_2_weights, depthwise_2_biases,
                    depthwise_2_output_mult, depthwise_2_output_shift,
                    input_size, weight_size, bias_size);

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims = {0}, output_dims;

    // Set input dimensions
    input_dims.n = 1;
    input_dims.h = input_h;
    input_dims.w = input_w;
    input_dims.c = input_c;

    // Set filter dimensions
    filter_dims.h = kernel_h;
    filter_dims.w = kernel_w;
    filter_dims.c = output_c;

    // Set output dimensions
    output_dims.h = output_h;
    output_dims.w = output_w;
    output_dims.c = output_c;

    // Set depthwise convolution parameters
    dw_conv_params.stride.h = 2;
    dw_conv_params.stride.w = 2;
    dw_conv_params.dilation.h = 1;
    dw_conv_params.dilation.w = 1;
    dw_conv_params.padding.h = 1;
    dw_conv_params.padding.w = 1;
    dw_conv_params.ch_mult = ch_mult;
    dw_conv_params.input_offset = 0;
    dw_conv_params.output_offset = 0;
    dw_conv_params.activation.min = -128;
    dw_conv_params.activation.max = 127;

    // Set quantization parameters
    quant_params.multiplier = depthwise_2_output_mult;
    quant_params.shift = depthwise_2_output_shift;

    // Calculate scratch buffer size
    const int32_t buf_size = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params,
                                                                           &input_dims,
                                                                           &filter_dims,
                                                                           &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();

    uint32_t start_cycles = read_cycle_counter();
    arm_depthwise_conv_wrapper_s8(&ctx,
                                  &dw_conv_params,
                                  &quant_params,
                                  &input_dims,
                                  depthwise_2_input,
                                  &filter_dims,
                                  depthwise_2_weights,
                                  &bias_dims,
                                  depthwise_2_biases,
                                  &output_dims,
                                  output);
    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;

    if (ctx.buf) {
    	memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    // Print results
    printf("layer1_arm_depthwise_conv_s8\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    *total_cycles += cycle_count;
    *total_stack += stack_used;

    // Free allocated memory
    free(depthwise_2_input);
    free(depthwise_2_weights);
    free(depthwise_2_biases);
    free(depthwise_2_output_mult);
    free(depthwise_2_output_shift);
    free(output);
}

void layer2_arm_depthwise_conv_s8(uint32_t *total_cycles, uint32_t *total_stack) {
    // Define tensor and layer parameters for Layer 2
    const int input_h = 25, input_w = 5, input_c = 64;
    const int kernel_h = 3, kernel_w = 3, ch_mult = 64;
    const int output_h = 25, output_w = 5, output_c = 64;

    const size_t input_size = input_h * input_w * input_c;
    const size_t weight_size = kernel_h * kernel_w * input_c * ch_mult;
    const size_t bias_size = output_c;
    const size_t output_size = output_h * output_w * output_c;

    // Allocate memory for inputs, weights, biases, and output
    int8_t *depthwise_2_input = malloc(input_size);
    int8_t *depthwise_2_weights = malloc(weight_size);
    int32_t *depthwise_2_biases = malloc(bias_size * sizeof(int32_t));
    int32_t *depthwise_2_output_mult = malloc(bias_size * sizeof(int32_t));
    int32_t *depthwise_2_output_shift = malloc(bias_size * sizeof(int32_t));
    int8_t *output = malloc(output_size);

    // Fill buffers with dummy data
    fill_dummy_data(depthwise_2_input, depthwise_2_weights, depthwise_2_biases,
                    depthwise_2_output_mult, depthwise_2_output_shift,
                    input_size, weight_size, bias_size);

    cmsis_nn_context ctx;
    cmsis_nn_dw_conv_params dw_conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims = {0}, output_dims;

    // Set input dimensions
    input_dims.n = 1;
    input_dims.h = input_h;
    input_dims.w = input_w;
    input_dims.c = input_c;

    // Set filter dimensions
    filter_dims.h = kernel_h;
    filter_dims.w = kernel_w;
    filter_dims.c = output_c;

    // Set output dimensions
    output_dims.h = output_h;
    output_dims.w = output_w;
    output_dims.c = output_c;

    // Set depthwise convolution parameters
    dw_conv_params.stride.h = 1;
    dw_conv_params.stride.w = 1;
    dw_conv_params.dilation.h = 1;
    dw_conv_params.dilation.w = 1;
    dw_conv_params.padding.h = 0;
    dw_conv_params.padding.w = 0;
    dw_conv_params.ch_mult = ch_mult;
    dw_conv_params.input_offset = 0;
    dw_conv_params.output_offset = 0;
    dw_conv_params.activation.min = -128;
    dw_conv_params.activation.max = 127;

    // Set quantization parameters
    quant_params.multiplier = depthwise_2_output_mult;
    quant_params.shift = depthwise_2_output_shift;

    // Calculate scratch buffer size
    const int32_t buf_size = arm_depthwise_conv_wrapper_s8_get_buffer_size(&dw_conv_params,
                                                                           &input_dims,
                                                                           &filter_dims,
                                                                           &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();

    uint32_t start_cycles = read_cycle_counter();
    arm_depthwise_conv_wrapper_s8(&ctx,
                                  &dw_conv_params,
                                  &quant_params,
                                  &input_dims,
                                  depthwise_2_input,
                                  &filter_dims,
                                  depthwise_2_weights,
                                  &bias_dims,
                                  depthwise_2_biases,
                                  &output_dims,
                                  output);
    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;

    if (ctx.buf) {
    	memset(ctx.buf, 0, buf_size);
        free(ctx.buf);
    }

    // Print results
    printf("layer2_arm_depthwise_conv_s8\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Layers 2, 4, 6, 8, 10
	cycle_count = cycle_count * 5;
	stack_used = stack_used * 5;

    printf("*****Layers 2, 4, 6, 8, 10: DEPTHWISE_CONV_2D cycle count: %lu\n\r", (unsigned long)cycle_count);
    printf("*****Layers 2, 4, 6, 8, 10: DEPTHWISE_CONV_2D stack usage: %lu\n\r", (unsigned long)stack_used);

    // Add to totals
    *total_cycles += cycle_count;
    *total_stack += stack_used;

    // Free allocated memory
    free(depthwise_2_input);
    free(depthwise_2_weights);
    free(depthwise_2_biases);
    free(depthwise_2_output_mult);
    free(depthwise_2_output_shift);
    free(output);
}

void layer3_arm_conv_s8(uint32_t *total_cycles, uint32_t *total_stack) {
    // Define tensor and layer parameters for Layer 3
    const int input_h = 25, input_w = 5, input_c = 64;
    const int kernel_h = 1, kernel_w = 1, ch_mult = 64;
    const int output_h = 25, output_w = 5, output_c = input_c;

    const size_t input_size = input_h * input_w * input_c;
    const size_t weight_size = kernel_h * kernel_w * input_c * ch_mult;
    const size_t bias_size = output_c;
    const size_t output_size = output_h * output_w * output_c;

    // Allocate memory for inputs, weights, biases, and output
    int8_t *conv_input = malloc(input_size);
    int8_t *conv_weights = malloc(weight_size);
    int32_t *conv_biases = malloc(bias_size * sizeof(int32_t));
    int32_t *conv_output_mult = malloc(bias_size * sizeof(int32_t));
    int32_t *conv_output_shift = malloc(bias_size * sizeof(int32_t));
    int8_t *output = malloc(output_size);

    // Fill buffers with dummy data
    fill_dummy_data(conv_input, conv_weights, conv_biases,
                    conv_output_mult, conv_output_shift,
                    input_size, weight_size, bias_size);

    cmsis_nn_context ctx;
    cmsis_nn_conv_params conv_params;
    cmsis_nn_per_channel_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims = {0}, output_dims;

    // Set input dimensions
    input_dims.n = 1;
    input_dims.h = input_h;
    input_dims.w = input_w;
    input_dims.c = input_c;

    // Set filter dimensions
    filter_dims.h = kernel_h;
    filter_dims.w = kernel_w;
    filter_dims.c = output_c;

    // Set output dimensions
    output_dims.h = output_h;
    output_dims.w = output_w;
    output_dims.c = output_c;

    // Set convolution parameters
    conv_params.stride.h = 1;
    conv_params.stride.w = 1;
    conv_params.dilation.h = 1;
    conv_params.dilation.w = 1;
    conv_params.padding.h = 1;
    conv_params.padding.w = 1;
    conv_params.input_offset = 0;
    conv_params.output_offset = 0;
    conv_params.activation.min = -128;
    conv_params.activation.max = 127;

    // Set quantization parameters
    quant_params.multiplier = conv_output_mult;
    quant_params.shift = conv_output_shift;

    // Calculate scratch buffer size
    const int32_t buf_size = arm_convolve_wrapper_s8_get_buffer_size(&conv_params,
                                                                     &input_dims,
                                                                     &filter_dims,
                                                                     &output_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();

    uint32_t start_cycles = read_cycle_counter();
    arm_convolve_wrapper_s8(&ctx,
                            &conv_params,
                            &quant_params,
                            &input_dims,
                            conv_input,
                            &filter_dims,
                            conv_weights,
                            &bias_dims,
                            conv_biases,
                            &output_dims,
                            output);
    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;

    if (ctx.buf) {
        free(ctx.buf);
    }

    // Print results
    printf("layer3_arm_conv_s8\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Layers 3, 5, 7, 9, 11
	cycle_count = cycle_count * 5;
	stack_used = stack_used * 5;

    printf("*****Layers 3, 5, 7, 9, 11: CONV_2D cycle count: %lu\n\r", (unsigned long)cycle_count);
    printf("*****Layers 3, 5, 7, 9, 11: CONV_2D stack usage: %lu\n\r", (unsigned long)stack_used);


    // Add to totals
    *total_cycles += cycle_count;
    *total_stack += stack_used;

    // Free allocated memory
    free(conv_input);
    free(conv_weights);
    free(conv_biases);
    free(conv_output_mult);
    free(conv_output_shift);
    free(output);
}

// Layer 12: AVERAGE_POOL_2D
void layer12_arm_avgpool_s8(uint32_t *total_cycles, uint32_t *total_stack) {
    // Define tensor and layer parameters for Layer 12
    const int input_h = 25, input_w = 5, input_c = 64;
    const int output_h = 1, output_w = 1, output_c = input_c;
    const int filter_h = 25, filter_w = 5;

    const size_t input_size = input_h * input_w * input_c;
    const size_t output_size = output_h * output_w * output_c;

    // Allocate memory for inputs and output
    int8_t *input = malloc(input_size);
    int8_t *output = malloc(output_size);

    // Fill input buffer with dummy data
    for (size_t i = 0; i < input_size; ++i) {
        input[i] = (rand() % 255) - 128; // Random int8_t
    }

    cmsis_nn_context ctx;
    cmsis_nn_pool_params pool_params;
    cmsis_nn_dims input_dims, output_dims,filter_dims;

    // Set input dimensions
    input_dims.n = 1;
    input_dims.h = input_h;
    input_dims.w = input_w;
    input_dims.c = input_c;

    filter_dims.w = filter_w;
    filter_dims.h = filter_h;

    // Set output dimensions
    output_dims.h = output_h;
    output_dims.w = output_w;
    output_dims.c = output_c;

    // Set pooling parameters
    pool_params.stride.h = 25;
    pool_params.stride.w = 5;
    pool_params.padding.h = 1;  // Padding is specified, adjust as needed
    pool_params.padding.w = 1;
    pool_params.activation.min = -128;
    pool_params.activation.max = 127;

    // Handle scratch buffer size (if needed by the pooling function)
    const int32_t buf_size = arm_avgpool_s8_get_buffer_size(output_w, input_c);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    // Performance measurement
    fill_stack_pattern_to_sp();
    enable_cycle_counter();

    uint32_t start_cycles = read_cycle_counter();
    arm_avgpool_s8(&ctx, &pool_params, &input_dims, input, &filter_dims, &output_dims, output);
    uint32_t end_cycles = read_cycle_counter();
    uint32_t stack_used = measure_stack_usage();
    uint32_t cycle_count = end_cycles - start_cycles;

    if (ctx.buf) {
        free(ctx.buf);
    }

    // Print results
    printf("layer12_arm_avgpool_s8\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    // Add to totals
    *total_cycles += cycle_count;
    *total_stack += stack_used;

    // Free allocated memory
    free(input);
    free(output);
}

// Layer 13: Fully Connected
void layer13_arm_fully_connected_s8(uint32_t *total_cycles, uint32_t *total_stack) {
    // Define tensor and layer parameters for Layer 13
    const int input_n = 1, input_h = 1, input_w = 1, input_c = 64;
    const int output_n = 1, output_c = 12;
    const int filter_n = 64, filter_c = 12;

    const size_t input_size = input_h * input_w * input_c;  // Input size (64)
    const size_t weight_size = filter_n * filter_c;         // Weights size (64 * 12)
    const size_t bias_size = output_c;                    // Bias size (12)
    const size_t output_size = output_n * output_c;                  // Output size (12)

    // Allocate memory for inputs, weights, biases, and output
    int8_t *input = malloc(input_size);
    int8_t *weights = malloc(weight_size);
    int32_t *bias = malloc(bias_size * sizeof(int32_t));
    int32_t *output_mult = malloc(bias_size * sizeof(int32_t));
    int32_t *output_shift = malloc(bias_size * sizeof(int32_t));
    int8_t *output = malloc(output_size);

    // Fill buffers with dummy data
    fill_dummy_data(input, weights, bias, output_mult, output_shift, input_size, weight_size, bias_size);

    cmsis_nn_context ctx;
    cmsis_nn_fc_params fc_params;
    cmsis_nn_per_tensor_quant_params quant_params;
    cmsis_nn_dims input_dims, filter_dims, bias_dims = {0}, output_dims;

    // Set input dimensions
    input_dims.n = input_n;
    input_dims.h = input_h;
    input_dims.w = input_w;
    input_dims.c = input_c;

    // Set filter dimensions
    filter_dims.n = filter_n;
    filter_dims.c = filter_c;

    // Set output dimensions
    output_dims.n = output_n;
    output_dims.c = output_c;

    // Set fully connected parameters
    fc_params.input_offset = 0;
    fc_params.filter_offset = 0;
    fc_params.output_offset = 0;
    fc_params.activation.min = -128;
    fc_params.activation.max = 127;

    // Set quantization parameters
    quant_params.multiplier = 1342580370;
    quant_params.shift = -9;

    // Handle scratch buffer size
    const int32_t buf_size = arm_fully_connected_s8_get_buffer_size(&filter_dims);
    ctx.buf = malloc(buf_size);
    ctx.size = buf_size;

    if (ctx.buf == NULL) {
        printf("Failed to allocate scratch buffer\n\r");
        free(input);
        free(weights);
        free(bias);
        free(output_mult);
        free(output_shift);
        free(output);
        return;
    }

    // Enable cycle counter for performance measurement
    enable_cycle_counter();

    // Fill stack with a known pattern
    fill_stack_pattern_to_sp();

    // Measure cycles
    uint32_t start_cycles = read_cycle_counter();
    arm_fully_connected_s8(&ctx,
    							&fc_params,
    							&quant_params,
    							&input_dims,
								input,
    							&filter_dims,
								weights,
    							&bias_dims,
								bias,
    							&output_dims,
    							output);
    uint32_t end_cycles = read_cycle_counter();

    // Measure stack usage
    uint32_t stack_used = measure_stack_usage();

    // Calculate cycle count
    uint32_t cycle_count = end_cycles - start_cycles;

    // Clear and free scratch buffer
    if (ctx.buf) {
        memset(ctx.buf, 0, ctx.size);
        free(ctx.buf);
    }

    // Print results
    printf("Layer 13: Fully Connected (arm_fully_connected_s8)\n\r");
    printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
    printf("Output:\n\r");
//    for (int i = 0; i < output_size; i++) {
//        printf("%d ", output[i]);
//    }
//    printf("\n\r");

    // Add to totals
    *total_cycles += cycle_count;
    *total_stack += stack_used;

    // Free allocated memory
    free(input);
    free(weights);
    free(bias);
    free(output_mult);
    free(output_shift);
    free(output);
}
