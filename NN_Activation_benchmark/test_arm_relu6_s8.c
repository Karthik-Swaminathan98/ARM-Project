#include <stdlib.h>
#include <arm_nnfunctions.h>
//#include "../Include/softmax/test_data.h"
#include "validate.h"

#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define RELU6_INPUT_SIZE 1024
//static int8_t relu6_s8_input[RELU6_INPUT_SIZE] = {};


static void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; // Enable DWT
    DWT->CYCCNT = 0;                                // Reset cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;            // Enable cycle counter
    DWT->CYCCNT = 0;
    DWT->CPICNT = 0;
    DWT->EXCCNT = 0;
    DWT->SLEEPCNT = 0;
    DWT->LSUCNT = 0;
    DWT->FOLDCNT = 0;
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

static void do_srand(void) {
    enable_cycle_counter();
    srand(read_cycle_counter());
}

static void generate_rand_s8(int8_t *src, int length) {
    do_srand();
    for (int i = 0; i < length; i++) {
        src[i] = (int8_t)((rand() % 256) - 128); // Range: [-128, 127]
    }
}

void relu6_arm_relu6_s8(void)
{
    const uint16_t input_sizes[] = {512, 1024, 2048};
    const int num_sizes = sizeof(input_sizes) / sizeof(input_sizes[0]);

    for (int i = 0; i < num_sizes; i++)
    {
        uint16_t size = input_sizes[i];

        // Allocate and initialize input buffer
        int8_t *input_data = (int8_t *)malloc(size * sizeof(int8_t));
        if (!input_data)
        {
            printf("Memory allocation failed for size %u\n\r", size);
            continue;
        }

        generate_rand_s8(input_data, size);

        fill_stack_pattern_to_sp();
        enable_cycle_counter();

        uint32_t start_cycles = read_cycle_counter();
        arm_relu6_s8(input_data, size);
        uint32_t end_cycles = read_cycle_counter();

        uint32_t cycle_count = end_cycles - start_cycles;
        uint32_t instr_est = cycle_count
                           - DWT->CPICNT
                           - DWT->EXCCNT
                           - DWT->SLEEPCNT
                           - DWT->LSUCNT
                           + DWT->FOLDCNT;
        uint32_t stack_used = measure_stack_usage();

        printf("Input Size: %u\n\r", size);
        printf("Cycle Count for arm_relu6_s8: %lu\n\r", (unsigned long)cycle_count);
        printf("Estimated Instruction Count: %lu\n", instr_est);
        printf("Stack Used for arm_relu6_s8: %lu bytes\n\r\n", (unsigned long)stack_used);

        free(input_data);
    }
}
