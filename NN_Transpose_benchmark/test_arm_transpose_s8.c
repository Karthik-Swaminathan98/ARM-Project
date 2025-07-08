#include "main.h"
#include "../TestData/transpose_3dim2/test_data.h"
#include "../TestData/transpose_matrix/test_data.h"
#include "TestData/transpose_default/test_data.h"
#define REPEAT_NUM (2)

RAM_FUNC void transpose_default_arm_transpose_s8(void)
{
    //const arm_cmsis_nn_status expected = ARM_CMSIS_NN_SUCCESS;
    int8_t output_data[TRANSPOSE_DEFAULT_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_DEFAULT_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_DEFAULT_OUT_DIM;

    const int8_t *input_data = transpose_default_input_tensor;
    const int8_t *const output_ref = transpose_default_output;
    const int32_t output_ref_size = TRANSPOSE_DEFAULT_SIZE;

    const uint32_t perm[TRANSPOSE_DEFAULT_PERM_SIZE] = TRANSPOSE_DEFAULT_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_DEFAULT_PERM_SIZE, perm};

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles = read_cycle_counter();

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

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

    printf("\n\r");
    if (validate(output_data, output_ref, output_ref_size)) {
		printf("arm_transpose_default_s8 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
	} else {
		printf("arm_transpose_default_s8 output validation FAILED\n\r");
	}
}

RAM_FUNC void transpose_3dim2_arm_transpose_s8(void)
{
    int8_t output_data[TRANSPOSE_3DIM2_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_3DIM2_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_3DIM2_OUT_DIM;

    const int8_t *input_data = transpose_3dim2_input_tensor;
    const int8_t *const output_ref = transpose_3dim2_output;
    const int32_t output_ref_size = TRANSPOSE_3DIM2_SIZE;

    const uint32_t perm[TRANSPOSE_3DIM2_PERM_SIZE] = TRANSPOSE_3DIM2_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_3DIM2_PERM_SIZE, perm};

    enable_cycle_counter();

    // Fill stack with known pattern
    fill_stack_pattern_to_sp();

    // Measure cycles
    uint32_t start_cycles = read_cycle_counter();

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

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

    printf("\n\r");
    if (validate(output_data, output_ref, output_ref_size)) {
        printf("arm_transpose_3dim2_s8 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    } else {
        printf("arm_transpose_3dim2_s8 output validation FAILED\n\r");
    }
}

RAM_FUNC void transpose_matrix_arm_transpose_s8(void)
{
    int8_t output_data[TRANSPOSE_MATRIX_SIZE] = {0};
    int8_t *output_ptr = output_data;

    const cmsis_nn_dims input_dims = TRANSPOSE_MATRIX_IN_DIM;
    const cmsis_nn_dims output_dims = TRANSPOSE_MATRIX_OUT_DIM;

    const int8_t *input_data = transpose_matrix_input_tensor;
    const int8_t *const output_ref = transpose_matrix_output;
    const int32_t output_ref_size = TRANSPOSE_MATRIX_SIZE;

    const uint32_t perm[TRANSPOSE_MATRIX_PERM_SIZE] = TRANSPOSE_MATRIX_PERM;
    const cmsis_nn_transpose_params transpose_params = {TRANSPOSE_MATRIX_PERM_SIZE, perm};

    enable_cycle_counter();
    fill_stack_pattern_to_sp();
    uint32_t start_cycles = read_cycle_counter();

    arm_transpose_s8(input_data, output_ptr, &input_dims, &output_dims, &transpose_params);

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

    printf("\n\r");
    if (validate(output_data, output_ref, output_ref_size)) {
        printf("arm_transpose_matrix_s8 output validation PASSED\n\r");
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n\r", instr_est);
        printf("Execution Time (approx): %.3f us\n\r", time_us);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);
    } else {
        printf("arm_transpose_matrix_s8 output validation FAILED\n\r");
    }
}

