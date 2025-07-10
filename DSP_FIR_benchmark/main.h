#ifndef MAIN_H
#define MAIN_H

#include "cyhal.h"
#include "cybsp.h"
#include "cyhal_clock.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

// Constants
#define TEST_LENGTH_SAMPLES  320
#define BLOCK_SIZE           32
#define NUM_TAPS_ARRAY_SIZE  29
#define NUM_TAPS             29
#define NUM_TAPS_q15             8
#define SNR_THRESHOLD_F32    75.0f

// Sine, step, ramp, and noise signal frequency and amplitude for generation
#define SINE_FREQ            1000.0f
#define SAMPLING_FREQ        48000.0f

// Attributes
#define RAM_FUNC __attribute__((section(".cy_ramfunc")))

// Stack Limit
extern uint32_t __StackLimit;

// Clock Frequency
extern uint32_t clkFastfreq;

extern const int FIR_SIZES[];
#define FIR_SIZES_COUNT 6
#define Q15_SCALE 32767
extern const float32_t firCoeffs32[NUM_TAPS_ARRAY_SIZE];
extern const q15_t firCoeffsQ15[NUM_TAPS_q15];

// Prototypes for common utilities
void fill_stack_pattern_to_sp(void);
void enable_cycle_counter(void);
uint32_t read_cycle_counter(void);
uint32_t measure_stack_usage(void);

#endif // MAIN_H
