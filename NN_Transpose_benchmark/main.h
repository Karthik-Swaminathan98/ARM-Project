#ifndef MAIN_H_
#define MAIN_H_

#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <arm_nnfunctions.h>
#include <arm_math.h>
#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "core_cm4.h"
#include "validate.h"

// RAM function attribute
#define RAM_FUNC __attribute__((section(".cy_ramfunc")))

// Stack Limit
extern uint32_t __StackLimit;

// Clock Frequency
extern uint32_t clkFastfreq;

// Function Prototypes
void enable_cycle_counter(void);
uint32_t read_cycle_counter(void);
void fill_stack_pattern_to_sp(void);
uint32_t measure_stack_usage(void);
void generate_rand_s8(int8_t *src, int length);
void generate_rand_s16(int16_t *src, int length);

#endif /* MAIN_H_ */
