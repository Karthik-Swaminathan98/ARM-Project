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
#define SINE_FREQ        50
#define SAMPLING_FREQ    256
#define FFT_SIZES_COUNT  6
#define Q15_SCALE        32768
#define NUM_EXECUTIONS   10

// Attributes
#define RAM_FUNC __attribute__((section(".cy_ramfunc")))

// Stack Limit
extern uint32_t __StackLimit;

// Clock Frequency
extern uint32_t clkFastfreq;

// FFT sizes to benchmark
extern const int FFT_SIZES[FFT_SIZES_COUNT];

// Prototypes for common utilities
void fill_stack_pattern_to_sp(void);
void enable_cycle_counter(void);
uint32_t read_cycle_counter(void);
uint32_t measure_stack_usage(void);
void calculate_averages(uint32_t* cycle_counts, uint32_t* instr_counts, float* exec_time_us, uint32_t* stack_usages, int num_executions);
void generate_sine_wave_f32(float32_t* input, int N, float signal_freq, float sampling_freq);
void generate_sine_wave_q15(q15_t* input, int N, float signal_freq, float sampling_freq);

#endif // MAIN_H
