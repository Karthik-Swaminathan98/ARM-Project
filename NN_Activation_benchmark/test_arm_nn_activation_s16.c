#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <arm_nnfunctions.h>
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"

#define LEFT_SHIFT 8
#define ACTIVATION_FUNC ARM_SIGMOID // You can change to ARM_TANH if needed

extern uint32_t __StackLimit;

// Enable and read cycle counter
static void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0;
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

static uint32_t read_cycle_counter() {
    return DWT->CYCCNT;
}

// Fill known pattern in stack
static void fill_stack_pattern_to_sp() {
    register uint32_t *sp;
    __asm volatile("mov %0, sp" : "=r"(sp));

    uint32_t *p = (uint32_t *)&__StackLimit;
    while (p < sp) {
        *p++ = 0xAAAAAAAA;
    }
}

// Measure used stack space
static uint32_t measure_stack_usage() {
    register uint32_t *sp;
    __asm volatile("mov %0, sp" : "=r"(sp));

    uint32_t *p = (uint32_t *)&__StackLimit;
    while (p < sp) {
        if (*p != 0xAAAAAAAA) {
            break;
        }
        p++;
    }

    return ((uint32_t)sp - (uint32_t)p);
}

// Random generator
static void do_srand(void) {
    enable_cycle_counter();
    srand(read_cycle_counter());
}

static void generate_rand_s16(int16_t *src, int length) {
    do_srand();
    for (int i = 0; i < length; i++) {
        src[i] = (int16_t)((rand() % 65536) - 32768); // Range: [-32768, 32767]
    }
}

void activ_arm_nn_activation_s16(void) {
    const int input_sizes[] = {512, 1024, 2048};
    const int num_sizes = sizeof(input_sizes) / sizeof(input_sizes[0]);

    for (int i = 0; i < num_sizes; i++) {
        int size = input_sizes[i];

        // Allocate buffers
        int16_t *input = (int16_t *)malloc(size * sizeof(int16_t));
        int16_t *output = (int16_t *)malloc(size * sizeof(int16_t));

        if (!input || !output) {
            printf("Memory allocation failed for input size %d\n\r", size);
            free(input);
            free(output);
            continue;
        }

        generate_rand_s16(input, size);
        memset(output, 0, size * sizeof(int16_t));

        enable_cycle_counter();
        fill_stack_pattern_to_sp();

        uint32_t start_cycles = read_cycle_counter();

        arm_nn_activation_s16(input, output, size, LEFT_SHIFT, ACTIVATION_FUNC);

        uint32_t end_cycles = read_cycle_counter();

        uint32_t cycle_count = end_cycles - start_cycles;
        uint32_t stack_used = measure_stack_usage();

        printf("Input Size: %d\n\r", size);
        printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("Stack Used: %lu bytes\n\r\n", (unsigned long)stack_used);

        free(input);
        free(output);
    }
}
