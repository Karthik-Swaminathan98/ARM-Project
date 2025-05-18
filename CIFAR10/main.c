/*
 * SPDX-FileCopyrightText: Copyright 2022-2025 Arm Limited and/or its affiliates
 * SPDX-License-Identifier: Apache-2.0
 */

#include "arm_nnfunctions.h"
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "core_cm4.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#include "arm_nnexamples_cifar10_parameter.h"
#include "arm_nnexamples_cifar10_weights.h"
#include "arm_nnexamples_cifar10_inputs.h"

// Buffers
#define MAX_BUFFER_SIZE 1024  // adjust as needed or compute via wrapper_get_buffer_size
static int8_t scratch_buffer[CONV1_OUT_DIM*CONV1_OUT_DIM*CONV1_OUT_CH +
                             POOL1_OUT_DIM*POOL1_OUT_DIM*CONV1_OUT_CH +
                             CONV2_OUT_DIM*CONV2_OUT_DIM*CONV2_OUT_CH +
                             POOL2_OUT_DIM*POOL2_OUT_DIM*CONV2_OUT_CH +
                             CONV3_OUT_DIM*CONV3_OUT_DIM*CONV3_OUT_CH +
                             POOL3_OUT_DIM*POOL3_OUT_DIM*CONV3_OUT_CH];
static int16_t col_buffer[CONV2_KER_DIM*CONV2_KER_DIM*CONV2_IM_CH*2];
static uint8_t image_data_buffer[CONV1_IM_DIM*CONV1_IM_DIM*CONV1_IM_CH] = IMG_DATA;
static int8_t output_data[IP1_OUT];

// Quantization params
static int32_t conv1_bias_s32[CONV1_OUT_CH];
static int32_t conv2_bias_s32[CONV2_OUT_CH];
static int32_t conv3_bias_s32[CONV3_OUT_CH];
static int32_t ip1_bias_s32[IP1_OUT];

static int32_t conv1_mult[CONV1_OUT_CH];
static int32_t conv1_shift[CONV1_OUT_CH];
static int32_t conv2_mult[CONV2_OUT_CH];
static int32_t conv2_shift[CONV2_OUT_CH];
static int32_t conv3_mult[CONV3_OUT_CH];
static int32_t conv3_shift[CONV3_OUT_CH];
static int32_t ip1_mult;
static int32_t ip1_shift;

void initialize_quant_params() {
    // biases
    for(int i=0;i<CONV1_OUT_CH;i++) {
        conv1_bias_s32[i] = (int32_t)CONV1_BIAS[i] << CONV1_BIAS_LSHIFT;
        conv1_mult[i]     = CONV1_OUT_MULT[i];
        conv1_shift[i]    = CONV1_OUT_RSHIFT[i];
    }
    for(int i=0;i<CONV2_OUT_CH;i++) {
        conv2_bias_s32[i] = (int32_t)CONV2_BIAS[i] << CONV2_BIAS_LSHIFT;
        conv2_mult[i]     = CONV2_OUT_MULT[i];
        conv2_shift[i]    = CONV2_OUT_RSHIFT[i];
    }
    for(int i=0;i<CONV3_OUT_CH;i++) {
        conv3_bias_s32[i] = (int32_t)CONV3_BIAS[i] << CONV3_BIAS_LSHIFT;
        conv3_mult[i]     = CONV3_OUT_MULT[i];
        conv3_shift[i]    = CONV3_OUT_RSHIFT[i];
    }
    for(int i=0;i<IP1_OUT;i++) {
        ip1_bias_s32[i] = (int32_t)IP1_BIAS[i] << IP1_BIAS_LSHIFT;
    }
    ip1_mult  = IP1_OUT_MULT;
    ip1_shift = IP1_OUT_RSHIFT;
}

void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

int main(void) {
    cy_rslt_t result;
    result = cybsp_init();
    if(result!=CY_RSLT_SUCCESS) CY_ASSERT(0);
    __enable_irq();
    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                                    CYBSP_DEBUG_UART_CTS, CYBSP_DEBUG_UART_RTS,
                                    CY_RETARGET_IO_BAUDRATE);
    if(result!=CY_RSLT_SUCCESS) CY_ASSERT(0);
    printf("CIFAR-10 inference start\r\n");

    initialize_quant_params();

    cmsis_nn_context ctx;
    ctx.buf  = col_buffer;
    ctx.size = sizeof(col_buffer);

    // pointers for double buffer
    int8_t *buf0 = scratch_buffer;
    int8_t *buf1 = scratch_buffer + (CONV1_OUT_DIM*CONV1_OUT_DIM*CONV1_OUT_CH);

    // preprocess
    for(int i=0, j=0;i<CONV1_IM_DIM*CONV1_IM_DIM*CONV1_IM_CH;i+=3,j+=3) {
        buf1[j]   = (int8_t)__SSAT(((int)image_data_buffer[j]   - INPUT_MEAN_SHIFT[0]) << INPUT_RIGHT_SHIFT[0],8);
        buf1[j+1] = (int8_t)__SSAT(((int)image_data_buffer[j+1] - INPUT_MEAN_SHIFT[1]) << INPUT_RIGHT_SHIFT[1],8);
        buf1[j+2] = (int8_t)__SSAT(((int)image_data_buffer[j+2] - INPUT_MEAN_SHIFT[2]) << INPUT_RIGHT_SHIFT[2],8);
    }

    cmsis_nn_conv_params conv_params = {0};
    cmsis_nn_per_channel_quant_params quant_ch = {0};
    cmsis_nn_dims dims_in, dims_filter, dims_bias, dims_out;
    cmsis_nn_pool_params pool_params = {0};
    cmsis_nn_fc_params fc_params = {0};
    cmsis_nn_per_tensor_quant_params quant_tensor = {0};

    // CONV1
    printf("CONV1...\r\n");
    conv_params.input_offset   = 0;
    conv_params.output_offset  = 0;
    conv_params.stride.h       = CONV1_STRIDE;
    conv_params.stride.w       = CONV1_STRIDE;
    conv_params.padding.h      = CONV1_PADDING;
    conv_params.padding.w      = CONV1_PADDING;
    conv_params.dilation.h     = 1;
    conv_params.dilation.w     = 1;
    conv_params.activation.min = -128;
    conv_params.activation.max = 127;

    quant_ch.multiplier = conv1_mult;
    quant_ch.shift      = conv1_shift;

    dims_in.n = 1;
    dims_in.h = CONV1_IM_DIM;
    dims_in.w = CONV1_IM_DIM;
    dims_in.c = CONV1_IM_CH;

    dims_filter.n = CONV1_OUT_CH;
    dims_filter.h = CONV1_KER_DIM;
    dims_filter.w = CONV1_KER_DIM;
    dims_filter.c = CONV1_IM_CH;

    dims_bias.c = CONV1_OUT_CH;

    dims_out.n = 1;
    dims_out.h = CONV1_OUT_DIM;
    dims_out.w = CONV1_OUT_DIM;
    dims_out.c = CONV1_OUT_CH;

    int8_t foo[256];

    arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_ch,
                            &dims_in, buf1,
                            &dims_filter, foo,
                            &dims_bias, conv1_bias_s32,
                            &dims_out, buf0);
    arm_relu6_s8(buf0, CONV1_OUT_DIM*CONV1_OUT_DIM*CONV1_OUT_CH);

    // POOL1
    printf("POOL1...\r\n");
    pool_params.stride.h       = POOL1_STRIDE;
    pool_params.stride.w       = POOL1_STRIDE;
    pool_params.padding.h      = POOL1_PADDING;
    pool_params.padding.w      = POOL1_PADDING;
    pool_params.activation.min = -128;
    pool_params.activation.max = 127;
    dims_in.h = CONV1_OUT_DIM; dims_in.w = CONV1_OUT_DIM; dims_in.c = CONV1_OUT_CH;
    dims_filter.h = POOL1_KER_DIM; dims_filter.w = POOL1_KER_DIM;
    dims_out.h = POOL1_OUT_DIM; dims_out.w = POOL1_OUT_DIM; dims_out.c = CONV1_OUT_CH;
    arm_max_pool_s8(&ctx, &pool_params, &dims_in, buf0, &dims_filter, &dims_out, buf1);

    // CONV2
    printf("CONV2...\r\n");
    conv_params.stride.h = CONV2_STRIDE;
    conv_params.stride.w = CONV2_STRIDE;
    conv_params.padding.h= CONV2_PADDING;
    conv_params.padding.w= CONV2_PADDING;
    quant_ch.multiplier = conv2_mult;
    quant_ch.shift      = conv2_shift;
    dims_in.h = POOL1_OUT_DIM; dims_in.w = POOL1_OUT_DIM; dims_in.c = CONV1_OUT_CH;
    dims_filter.n = CONV2_OUT_CH; dims_filter.h = CONV2_KER_DIM; dims_filter.w = CONV2_KER_DIM; dims_filter.c = CONV1_OUT_CH;
    dims_bias.c = CONV2_OUT_CH;
    dims_out.h = CONV2_OUT_DIM; dims_out.w = CONV2_OUT_DIM; dims_out.c = CONV2_OUT_CH;
    arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_ch,
                            &dims_in, buf1,
                            &dims_filter, conv2_wt,
                            &dims_bias, conv2_bias_s32,
                            &dims_out, buf0);
    arm_relu6_s8(buf0, CONV2_OUT_DIM*CONV2_OUT_DIM*CONV2_OUT_CH);

    // POOL2
    printf("POOL2...\r\n");
    pool_params.stride.h = POOL2_STRIDE;
    pool_params.stride.w = POOL2_STRIDE;
    pool_params.padding.h= POOL2_PADDING;
    pool_params.padding.w= POOL2_PADDING;
    dims_in.h = CONV2_OUT_DIM; dims_in.w = CONV2_OUT_DIM; dims_in.c = CONV2_OUT_CH;
    dims_filter.h = POOL2_KER_DIM; dims_filter.w = POOL2_KER_DIM;
    dims_out.h = POOL2_OUT_DIM; dims_out.w = POOL2_OUT_DIM; dims_out.c = CONV2_OUT_CH;
    arm_max_pool_s8(&ctx, &pool_params, &dims_in, buf0, &dims_filter, &dims_out, buf1);

    // CONV3
    printf("CONV3...\r\n");
    conv_params.stride.h = CONV3_STRIDE;
    conv_params.stride.w = CONV3_STRIDE;
    conv_params.padding.h= CONV3_PADDING;
    conv_params.padding.w= CONV3_PADDING;
    quant_ch.multiplier = conv3_mult;
    quant_ch.shift      = conv3_shift;
    dims_in.h = POOL2_OUT_DIM; dims_in.w = POOL2_OUT_DIM; dims_in.c = CONV2_OUT_CH;
    dims_filter.n = CONV3_OUT_CH; dims_filter.h = CONV3_KER_DIM; dims_filter.w = CONV3_KER_DIM; dims_filter.c = CONV2_OUT_CH;
    dims_bias.c = CONV3_OUT_CH;
    dims_out.h = CONV3_OUT_DIM; dims_out.w = CONV3_OUT_DIM; dims_out.c = CONV3_OUT_CH;
    arm_convolve_wrapper_s8(&ctx, &conv_params, &quant_ch,
                            &dims_in, buf1,
                            &dims_filter, conv3_wt,
                            &dims_bias, conv3_bias_s32,
                            &dims_out, buf0);
    arm_relu6_s8(buf0, CONV3_OUT_DIM*CONV3_OUT_DIM*CONV3_OUT_CH);

    // POOL3
    printf("POOL3...\r\n");
    pool_params.stride.h = POOL3_STRIDE;
    pool_params.stride.w = POOL3_STRIDE;
    pool_params.padding.h= POOL3_PADDING;
    pool_params.padding.w= POOL3_PADDING;
    dims_in.h = CONV3_OUT_DIM; dims_in.w = CONV3_OUT_DIM; dims_in.c = CONV3_OUT_CH;
    dims_filter.h = POOL3_KER_DIM; dims_filter.w = POOL3_KER_DIM;
    dims_out.h = POOL3_OUT_DIM; dims_out.w = POOL3_OUT_DIM; dims_out.c = CONV3_OUT_CH;
    arm_max_pool_s8(&ctx, &pool_params, &dims_in, buf0, &dims_filter, &dims_out, buf1);

    // Fully-connected
    printf("FC...\r\n");
    fc_params.input_offset  = -INPUT_ZP;
    fc_params.filter_offset = -WEIGHT_ZP;
    fc_params.output_offset = OUTPUT_ZP;
    fc_params.activation.min= -128;
    fc_params.activation.max= 127;
    quant_tensor.multiplier = ip1_mult;
    quant_tensor.shift      = ip1_shift;
    dims_in.n = 1; dims_in.h = 1; dims_in.w = 1; dims_in.c = IP1_DIM;
    dims_filter.n = IP1_OUT; dims_filter.h = 1; dims_filter.w = IP1_DIM; dims_filter.c = 1;
    dims_bias.c = IP1_OUT;
    dims_out.n = 1; dims_out.h = 1; dims_out.w = 1; dims_out.c = IP1_OUT;
    arm_fully_connected_s8(&ctx, &fc_params, &quant_tensor,
                           &dims_in, buf1,
                           &dims_filter, ip1_wt,
                           &dims_bias, ip1_bias_s32,
                           &dims_out, output_data);

    // Softmax
    printf("Softmax...\r\n");
    arm_softmax_s8(output_data,
                   1,
                   IP1_OUT,
                   SOFTMAX_SCALE_MULT,
                   SOFTMAX_SHIFT,
                   SOFTMAX_DIFF_MIN,
                   output_data);

    // print
    printf("Results:\r\n");
    for(int i=0;i<IP1_OUT;i++) {
        printf("%d: %d\r\n", i, output_data[i]);
    }
    printf("Done.\r\n");

    return 0;
}
