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
#include <stdbool.h>
#include <stdio.h>
#include "TestData/lstm_1/test_data.h"
#include "../TestData/lstm_2/test_data.h"
#include "../TestData/lstm_one_time_step/test_data.h"

// update the buffer size if adding a unit test with larger buffer.
#define LARGEST_BUFFER_SIZE LSTM_1_HIDDEN_SIZE *LSTM_1_BATCH_SIZE *LSTM_1_TIME_STEPS

static int8_t buffer1[LARGEST_BUFFER_SIZE];
static int8_t buffer2[LARGEST_BUFFER_SIZE];
static int8_t buffer3[LARGEST_BUFFER_SIZE];


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

void lstm_1(void)
{
    int8_t output[LSTM_1_BATCH_SIZE * LSTM_1_TIME_STEPS * LSTM_1_HIDDEN_SIZE] = {0};
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *output_ref = &lstm_1_output[0];
    const int32_t output_ref_size = LSTM_1_BATCH_SIZE * LSTM_1_TIME_STEPS * LSTM_1_HIDDEN_SIZE;

    int32_t input_data_kernel_sum[LSTM_1_HIDDEN_SIZE];
    int32_t forget_data_kernel_sum[LSTM_1_HIDDEN_SIZE];
    int32_t cell_data_kernel_sum[LSTM_1_HIDDEN_SIZE];
    int32_t output_data_kernel_sum[LSTM_1_HIDDEN_SIZE];

    int32_t input_hidden_kernel_sum[LSTM_1_HIDDEN_SIZE];
    int32_t forget_hidden_kernel_sum[LSTM_1_HIDDEN_SIZE];
    int32_t cell_hidden_kernel_sum[LSTM_1_HIDDEN_SIZE];
    int32_t output_hidden_kernel_sum[LSTM_1_HIDDEN_SIZE];

    arm_vector_sum_s8(&input_data_kernel_sum[0],
                      LSTM_1_INPUT_SIZE,
                      LSTM_1_HIDDEN_SIZE,
                      &lstm_1_input_gate_input_weights[0],
                      LSTM_1_INPUT_ZERO_POINT,
                      0,
                      &lstm_1_input_gate_bias[0]);
    arm_vector_sum_s8(&forget_data_kernel_sum[0],
                      LSTM_1_INPUT_SIZE,
                      LSTM_1_HIDDEN_SIZE,
                      &lstm_1_forget_gate_input_weights[0],
                      LSTM_1_INPUT_ZERO_POINT,
                      0,
                      &lstm_1_forget_gate_bias[0]);
    arm_vector_sum_s8(&cell_data_kernel_sum[0],
                      LSTM_1_INPUT_SIZE,
                      LSTM_1_HIDDEN_SIZE,
                      &lstm_1_cell_gate_input_weights[0],
                      LSTM_1_INPUT_ZERO_POINT,
                      0,
                      &lstm_1_cell_gate_bias[0]);
    arm_vector_sum_s8(&output_data_kernel_sum[0],
                      LSTM_1_INPUT_SIZE,
                      LSTM_1_HIDDEN_SIZE,
                      &lstm_1_output_gate_input_weights[0],
                      LSTM_1_INPUT_ZERO_POINT,
                      0,
                      &lstm_1_output_gate_bias[0]);

    arm_vector_sum_s8(&input_hidden_kernel_sum[0],
                      LSTM_1_HIDDEN_SIZE,
                      LSTM_1_HIDDEN_SIZE,
                      &lstm_1_input_gate_hidden_weights[0],
                      -LSTM_1_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&forget_hidden_kernel_sum[0],
                      LSTM_1_HIDDEN_SIZE,
                      LSTM_1_HIDDEN_SIZE,
                      &lstm_1_forget_gate_hidden_weights[0],
                      -LSTM_1_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&cell_hidden_kernel_sum[0],
                      LSTM_1_HIDDEN_SIZE,
                      LSTM_1_HIDDEN_SIZE,
                      &lstm_1_cell_gate_hidden_weights[0],
                      -LSTM_1_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&output_hidden_kernel_sum[0],
                      LSTM_1_HIDDEN_SIZE,
                      LSTM_1_HIDDEN_SIZE,
                      &lstm_1_output_gate_hidden_weights[0],
                      -LSTM_1_OUTPUT_ZERO_POINT,
                      0,
                      NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_1_INPUT_GATE_INPUT_MULTIPLIER,
                                           LSTM_1_INPUT_GATE_INPUT_SHIFT,
                                           &lstm_1_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_1_INPUT_GATE_HIDDEN_MULTIPLIER,
                                           LSTM_1_INPUT_GATE_HIDDEN_SHIFT,
                                           &lstm_1_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_1_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_1_FORGET_GATE_INPUT_MULTIPLIER,
                                            LSTM_1_FORGET_GATE_INPUT_SHIFT,
                                            &lstm_1_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_1_FORGET_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_1_FORGET_GATE_HIDDEN_SHIFT,
                                            &lstm_1_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_1_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_1_CELL_GATE_INPUT_MULTIPLIER,
                                          LSTM_1_CELL_GATE_INPUT_SHIFT,
                                          &lstm_1_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_1_CELL_GATE_HIDDEN_MULTIPLIER,
                                          LSTM_1_CELL_GATE_HIDDEN_SHIFT,
                                          &lstm_1_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_1_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_1_OUTPUT_GATE_INPUT_MULTIPLIER,
                                            LSTM_1_OUTPUT_GATE_INPUT_SHIFT,
                                            &lstm_1_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_1_OUTPUT_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_1_OUTPUT_GATE_HIDDEN_SHIFT,
                                            &lstm_1_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_1_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_1_TIME_MAJOR,
                                         LSTM_1_BATCH_SIZE,
                                         LSTM_1_TIME_STEPS,
                                         LSTM_1_INPUT_SIZE,
                                         LSTM_1_HIDDEN_SIZE,
                                         LSTM_1_INPUT_ZERO_POINT,
                                         LSTM_1_FORGET_TO_CELL_MULTIPLIER,
                                         LSTM_1_FORGET_TO_CELL_SHIFT,
                                         LSTM_1_INPUT_TO_CELL_MULTIPLIER,
                                         LSTM_1_INPUT_TO_CELL_SHIFT,
                                         LSTM_1_CELL_CLIP,
                                         LSTM_1_CELL_SCALE_POWER,
                                         LSTM_1_OUTPUT_MULTIPLIER,
                                         LSTM_1_OUTPUT_SHIFT,
                                         LSTM_1_OUTPUT_ZERO_POINT,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    enable_cycle_counter();

	// Fill stack with a known pattern
	fill_stack_pattern_to_sp();

	// Measure cycles
	uint32_t start_cycles_s8 = read_cycle_counter();

    arm_lstm_unidirectional_s8(lstm_1_input_tensor, output, &params, &buffers);

    // Measure cycles
    uint32_t end_cycles_s8 = read_cycle_counter();

    // Measure stack usage
    uint32_t stack_used_s8 = measure_stack_usage();

    // Calculate cycle count
    uint32_t cycle_count_s8 = end_cycles_s8 - start_cycles_s8;
    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
		printf("LSTM_1_S8 output validation PASSED\n\r");
		printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used_s8);
		printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count_s8);
	} else {
		printf("LSTM_1_S8 output validation FAILED\n\r");
	}

}

void lstm_2(void)
{
    int8_t output[LSTM_2_BATCH_SIZE * LSTM_2_TIME_STEPS * LSTM_2_HIDDEN_SIZE] = {0};
    const int8_t *output_ref = &lstm_2_output[0];
    const int32_t output_ref_size = LSTM_2_BATCH_SIZE * LSTM_2_TIME_STEPS * LSTM_2_HIDDEN_SIZE;

    int32_t input_data_kernel_sum[LSTM_2_HIDDEN_SIZE];
    int32_t forget_data_kernel_sum[LSTM_2_HIDDEN_SIZE];
    int32_t cell_data_kernel_sum[LSTM_2_HIDDEN_SIZE];
    int32_t output_data_kernel_sum[LSTM_2_HIDDEN_SIZE];

    int32_t input_hidden_kernel_sum[LSTM_2_HIDDEN_SIZE];
    int32_t forget_hidden_kernel_sum[LSTM_2_HIDDEN_SIZE];
    int32_t cell_hidden_kernel_sum[LSTM_2_HIDDEN_SIZE];
    int32_t output_hidden_kernel_sum[LSTM_2_HIDDEN_SIZE];

    arm_vector_sum_s8(&input_data_kernel_sum[0],
                      LSTM_2_INPUT_SIZE,
                      LSTM_2_HIDDEN_SIZE,
                      &lstm_2_input_gate_input_weights[0],
                      LSTM_2_INPUT_ZERO_POINT,
                      0,
                      &lstm_2_input_gate_bias[0]);
    arm_vector_sum_s8(&forget_data_kernel_sum[0],
                      LSTM_2_INPUT_SIZE,
                      LSTM_2_HIDDEN_SIZE,
                      &lstm_2_forget_gate_input_weights[0],
                      LSTM_2_INPUT_ZERO_POINT,
                      0,
                      &lstm_2_forget_gate_bias[0]);
    arm_vector_sum_s8(&cell_data_kernel_sum[0],
                      LSTM_2_INPUT_SIZE,
                      LSTM_2_HIDDEN_SIZE,
                      &lstm_2_cell_gate_input_weights[0],
                      LSTM_2_INPUT_ZERO_POINT,
                      0,
                      &lstm_2_cell_gate_bias[0]);
    arm_vector_sum_s8(&output_data_kernel_sum[0],
                      LSTM_2_INPUT_SIZE,
                      LSTM_2_HIDDEN_SIZE,
                      &lstm_2_output_gate_input_weights[0],
                      LSTM_2_INPUT_ZERO_POINT,
                      0,
                      &lstm_2_output_gate_bias[0]);

    arm_vector_sum_s8(&input_hidden_kernel_sum[0],
                      LSTM_2_HIDDEN_SIZE,
                      LSTM_2_HIDDEN_SIZE,
                      &lstm_2_input_gate_hidden_weights[0],
                      -LSTM_2_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&forget_hidden_kernel_sum[0],
                      LSTM_2_HIDDEN_SIZE,
                      LSTM_2_HIDDEN_SIZE,
                      &lstm_2_forget_gate_hidden_weights[0],
                      -LSTM_2_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&cell_hidden_kernel_sum[0],
                      LSTM_2_HIDDEN_SIZE,
                      LSTM_2_HIDDEN_SIZE,
                      &lstm_2_cell_gate_hidden_weights[0],
                      -LSTM_2_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&output_hidden_kernel_sum[0],
                      LSTM_2_HIDDEN_SIZE,
                      LSTM_2_HIDDEN_SIZE,
                      &lstm_2_output_gate_hidden_weights[0],
                      -LSTM_2_OUTPUT_ZERO_POINT,
                      0,
                      NULL);

    const cmsis_nn_lstm_gate gate_input = {LSTM_2_INPUT_GATE_INPUT_MULTIPLIER,
                                           LSTM_2_INPUT_GATE_INPUT_SHIFT,
                                           &lstm_2_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_2_INPUT_GATE_HIDDEN_MULTIPLIER,
                                           LSTM_2_INPUT_GATE_HIDDEN_SHIFT,
                                           &lstm_2_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_2_input_gate_bias[0],
                                           ARM_SIGMOID};

    const cmsis_nn_lstm_gate gate_forget = {LSTM_2_FORGET_GATE_INPUT_MULTIPLIER,
                                            LSTM_2_FORGET_GATE_INPUT_SHIFT,
                                            &lstm_2_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_2_FORGET_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_2_FORGET_GATE_HIDDEN_SHIFT,
                                            &lstm_2_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_2_forget_gate_bias[0],
                                            ARM_SIGMOID};

    const cmsis_nn_lstm_gate gate_cell = {LSTM_2_CELL_GATE_INPUT_MULTIPLIER,
                                          LSTM_2_CELL_GATE_INPUT_SHIFT,
                                          &lstm_2_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_2_CELL_GATE_HIDDEN_MULTIPLIER,
                                          LSTM_2_CELL_GATE_HIDDEN_SHIFT,
                                          &lstm_2_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_2_cell_gate_bias[0],
                                          ARM_TANH};

    const cmsis_nn_lstm_gate gate_output = {LSTM_2_OUTPUT_GATE_INPUT_MULTIPLIER,
                                            LSTM_2_OUTPUT_GATE_INPUT_SHIFT,
                                            &lstm_2_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_2_OUTPUT_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_2_OUTPUT_GATE_HIDDEN_SHIFT,
                                            &lstm_2_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_2_output_gate_bias[0],
                                            ARM_SIGMOID};

    const cmsis_nn_lstm_params params = {LSTM_2_TIME_MAJOR,
                                         LSTM_2_BATCH_SIZE,
                                         LSTM_2_TIME_STEPS,
                                         LSTM_2_INPUT_SIZE,
                                         LSTM_2_HIDDEN_SIZE,
                                         LSTM_2_INPUT_ZERO_POINT,
                                         LSTM_2_FORGET_TO_CELL_MULTIPLIER,
                                         LSTM_2_FORGET_TO_CELL_SHIFT,
                                         LSTM_2_INPUT_TO_CELL_MULTIPLIER,
                                         LSTM_2_INPUT_TO_CELL_SHIFT,
                                         LSTM_2_CELL_CLIP,
                                         LSTM_2_CELL_SCALE_POWER,
                                         LSTM_2_OUTPUT_MULTIPLIER,
                                         LSTM_2_OUTPUT_SHIFT,
                                         LSTM_2_OUTPUT_ZERO_POINT,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    enable_cycle_counter();

    // Fill stack with known pattern
    fill_stack_pattern_to_sp();

    // Measure cycles
    uint32_t start_cycles_s8 = read_cycle_counter();

    arm_lstm_unidirectional_s8(lstm_2_input_tensor, output, &params, &buffers);

    uint32_t end_cycles_s8 = read_cycle_counter();
    uint32_t stack_used_s8 = measure_stack_usage();
    uint32_t cycle_count_s8 = end_cycles_s8 - start_cycles_s8;
    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
        printf("LSTM_2_S8 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used_s8);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count_s8);
    } else {
        printf("LSTM_2_S8 output validation FAILED\n\r");
    }
}

void lstm_one_time_step(void)
{
    int8_t output[LSTM_ONE_TIME_STEP_BATCH_SIZE * LSTM_ONE_TIME_STEP_TIME_STEPS * LSTM_ONE_TIME_STEP_HIDDEN_SIZE] = {0};
   // const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    const int8_t *output_ref = &lstm_one_time_step_output[0];
    const int32_t output_ref_size =
        LSTM_ONE_TIME_STEP_BATCH_SIZE * LSTM_ONE_TIME_STEP_TIME_STEPS * LSTM_ONE_TIME_STEP_HIDDEN_SIZE;

    int32_t input_data_kernel_sum[LSTM_ONE_TIME_STEP_HIDDEN_SIZE];
    int32_t forget_data_kernel_sum[LSTM_ONE_TIME_STEP_HIDDEN_SIZE];
    int32_t cell_data_kernel_sum[LSTM_ONE_TIME_STEP_HIDDEN_SIZE];
    int32_t output_data_kernel_sum[LSTM_ONE_TIME_STEP_HIDDEN_SIZE];

    int32_t input_hidden_kernel_sum[LSTM_ONE_TIME_STEP_HIDDEN_SIZE];
    int32_t forget_hidden_kernel_sum[LSTM_ONE_TIME_STEP_HIDDEN_SIZE];
    int32_t cell_hidden_kernel_sum[LSTM_ONE_TIME_STEP_HIDDEN_SIZE];
    int32_t output_hidden_kernel_sum[LSTM_ONE_TIME_STEP_HIDDEN_SIZE];

    arm_vector_sum_s8(&input_data_kernel_sum[0],
                      LSTM_ONE_TIME_STEP_INPUT_SIZE,
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      &lstm_one_time_step_input_gate_input_weights[0],
                      LSTM_ONE_TIME_STEP_INPUT_ZERO_POINT,
                      0,
                      &lstm_one_time_step_input_gate_bias[0]);
    arm_vector_sum_s8(&forget_data_kernel_sum[0],
                      LSTM_ONE_TIME_STEP_INPUT_SIZE,
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      &lstm_one_time_step_forget_gate_input_weights[0],
                      LSTM_ONE_TIME_STEP_INPUT_ZERO_POINT,
                      0,
                      &lstm_one_time_step_forget_gate_bias[0]);
    arm_vector_sum_s8(&cell_data_kernel_sum[0],
                      LSTM_ONE_TIME_STEP_INPUT_SIZE,
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      &lstm_one_time_step_cell_gate_input_weights[0],
                      LSTM_ONE_TIME_STEP_INPUT_ZERO_POINT,
                      0,
                      &lstm_one_time_step_cell_gate_bias[0]);
    arm_vector_sum_s8(&output_data_kernel_sum[0],
                      LSTM_ONE_TIME_STEP_INPUT_SIZE,
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      &lstm_one_time_step_output_gate_input_weights[0],
                      LSTM_ONE_TIME_STEP_INPUT_ZERO_POINT,
                      0,
                      &lstm_one_time_step_output_gate_bias[0]);

    arm_vector_sum_s8(&input_hidden_kernel_sum[0],
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      &lstm_one_time_step_input_gate_hidden_weights[0],
                      -LSTM_ONE_TIME_STEP_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&forget_hidden_kernel_sum[0],
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      &lstm_one_time_step_forget_gate_hidden_weights[0],
                      -LSTM_ONE_TIME_STEP_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&cell_hidden_kernel_sum[0],
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      &lstm_one_time_step_cell_gate_hidden_weights[0],
                      -LSTM_ONE_TIME_STEP_OUTPUT_ZERO_POINT,
                      0,
                      NULL);
    arm_vector_sum_s8(&output_hidden_kernel_sum[0],
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                      &lstm_one_time_step_output_gate_hidden_weights[0],
                      -LSTM_ONE_TIME_STEP_OUTPUT_ZERO_POINT,
                      0,
                      NULL);

    // INPUT GATE
    const cmsis_nn_lstm_gate gate_input = {LSTM_ONE_TIME_STEP_INPUT_GATE_INPUT_MULTIPLIER,
                                           LSTM_ONE_TIME_STEP_INPUT_GATE_INPUT_SHIFT,
                                           &lstm_one_time_step_input_gate_input_weights[0],
                                           &input_data_kernel_sum[0],
                                           LSTM_ONE_TIME_STEP_INPUT_GATE_HIDDEN_MULTIPLIER,
                                           LSTM_ONE_TIME_STEP_INPUT_GATE_HIDDEN_SHIFT,
                                           &lstm_one_time_step_input_gate_hidden_weights[0],
                                           &input_hidden_kernel_sum[0],
                                           &lstm_one_time_step_input_gate_bias[0],
                                           ARM_SIGMOID};

    // FORGET GATE
    const cmsis_nn_lstm_gate gate_forget = {LSTM_ONE_TIME_STEP_FORGET_GATE_INPUT_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_FORGET_GATE_INPUT_SHIFT,
                                            &lstm_one_time_step_forget_gate_input_weights[0],
                                            &forget_data_kernel_sum[0],
                                            LSTM_ONE_TIME_STEP_FORGET_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_FORGET_GATE_HIDDEN_SHIFT,
                                            &lstm_one_time_step_forget_gate_hidden_weights[0],
                                            &forget_hidden_kernel_sum[0],
                                            &lstm_one_time_step_forget_gate_bias[0],
                                            ARM_SIGMOID};

    // CELL GATE
    const cmsis_nn_lstm_gate gate_cell = {LSTM_ONE_TIME_STEP_CELL_GATE_INPUT_MULTIPLIER,
                                          LSTM_ONE_TIME_STEP_CELL_GATE_INPUT_SHIFT,
                                          &lstm_one_time_step_cell_gate_input_weights[0],
                                          &cell_data_kernel_sum[0],
                                          LSTM_ONE_TIME_STEP_CELL_GATE_HIDDEN_MULTIPLIER,
                                          LSTM_ONE_TIME_STEP_CELL_GATE_HIDDEN_SHIFT,
                                          &lstm_one_time_step_cell_gate_hidden_weights[0],
                                          &cell_hidden_kernel_sum[0],
                                          &lstm_one_time_step_cell_gate_bias[0],
                                          ARM_TANH};

    // OUTPUT GATE
    const cmsis_nn_lstm_gate gate_output = {LSTM_ONE_TIME_STEP_OUTPUT_GATE_INPUT_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_OUTPUT_GATE_INPUT_SHIFT,
                                            &lstm_one_time_step_output_gate_input_weights[0],
                                            &output_data_kernel_sum[0],
                                            LSTM_ONE_TIME_STEP_OUTPUT_GATE_HIDDEN_MULTIPLIER,
                                            LSTM_ONE_TIME_STEP_OUTPUT_GATE_HIDDEN_SHIFT,
                                            &lstm_one_time_step_output_gate_hidden_weights[0],
                                            &output_hidden_kernel_sum[0],
                                            &lstm_one_time_step_output_gate_bias[0],
                                            ARM_SIGMOID};

    // LSTM DATA
    const cmsis_nn_lstm_params params = {LSTM_ONE_TIME_STEP_TIME_MAJOR,
                                         LSTM_ONE_TIME_STEP_BATCH_SIZE,
                                         LSTM_ONE_TIME_STEP_TIME_STEPS,
                                         LSTM_ONE_TIME_STEP_INPUT_SIZE,
                                         LSTM_ONE_TIME_STEP_HIDDEN_SIZE,
                                         LSTM_ONE_TIME_STEP_INPUT_ZERO_POINT,
                                         LSTM_ONE_TIME_STEP_FORGET_TO_CELL_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_FORGET_TO_CELL_SHIFT,
                                         LSTM_ONE_TIME_STEP_INPUT_TO_CELL_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_INPUT_TO_CELL_SHIFT,
                                         LSTM_ONE_TIME_STEP_CELL_CLIP,
                                         LSTM_ONE_TIME_STEP_CELL_SCALE_POWER,
                                         LSTM_ONE_TIME_STEP_OUTPUT_MULTIPLIER,
                                         LSTM_ONE_TIME_STEP_OUTPUT_SHIFT,
                                         LSTM_ONE_TIME_STEP_OUTPUT_ZERO_POINT,
                                         gate_forget,
                                         gate_input,
                                         gate_cell,
                                         gate_output};

    cmsis_nn_lstm_context buffers;
    buffers.temp1 = buffer1;
    buffers.temp2 = buffer2;
    buffers.cell_state = buffer3;

    enable_cycle_counter();

    // Fill stack with known pattern
    fill_stack_pattern_to_sp();

    // Measure cycles
    uint32_t start_cycles_s8 = read_cycle_counter();

    arm_lstm_unidirectional_s8(lstm_one_time_step_input_tensor, output, &params, &buffers);

    uint32_t end_cycles_s8 = read_cycle_counter();
    uint32_t stack_used_s8 = measure_stack_usage();
    uint32_t cycle_count_s8 = end_cycles_s8 - start_cycles_s8;
    printf("\n\r");
    if (validate(output, output_ref, output_ref_size)) {
        printf("LSTM_ONE_TIME_STEP_S8 output validation PASSED\n\r");
        printf("Stack Used: %lu bytes\n\r", (unsigned long)stack_used_s8);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count_s8);
    } else {
        printf("LSTM_ONE_TIME_STEP_S8 output validation FAILED\n\r");
    }
}


