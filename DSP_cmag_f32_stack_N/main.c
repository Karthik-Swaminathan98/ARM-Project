#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define SINE_FREQ 50           // Frequency of the sine wave
#define SAMPLING_FREQ 256      // Sampling frequency
#define FFT_SIZES_COUNT 6      // Total number of FFT sizes to test

// Array of FFT sizes to test
const int FFT_SIZES[FFT_SIZES_COUNT] = {32, 64, 128, 256, 512, 1024};

extern uint32_t __StackLimit;
extern uint32_t __StackTop;

void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; // Enable DWT
    DWT->CYCCNT = 0;                                // Reset cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;            // Enable cycle counter
}

uint32_t read_cycle_counter() {
    return DWT->CYCCNT;
}

void fill_stack_pattern_to_sp() {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));

    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) {
        *p++ = 0xAAAAAAAA;
    }
}

uint32_t measure_stack_usage() {
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

void generate_sine_wave_f32(float32_t* input, int N, float signal_freq, float sampling_freq) {
    for (int i = 0; i < N; i++) {
        float32_t value = sinf(2 * M_PI * signal_freq * i / sampling_freq);
        input[2 * i] = value;  // Real part
        input[2 * i + 1] = 0;  // Imaginary part (set to 0)
    }
}

int main(void) {
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

    printf("****************** \n\r");
    printf("Starting FFT Stack Usage Benchmark Program (f32)\n\r");

    enable_cycle_counter();

    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        // Allocate memory
        float32_t* input = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* output = (float32_t*)malloc(N * sizeof(float32_t));

        if (input == NULL || output == NULL) {
            printf("Memory allocation failed for FFT size N = %d\n\r", N);
            return -1;
        }

        // Generate 50 Hz sine wave
        generate_sine_wave_f32(input, N, SINE_FREQ, SAMPLING_FREQ);

        // Create FFT instance
        arm_cfft_instance_f32 fft_instance;
        if (arm_cfft_init_f32(&fft_instance, N) != ARM_MATH_SUCCESS) {
            printf("FFT initialization failed for N = %d\n\r", N);
            free(input);
            free(output);
            continue;
        }

        printf("FFT Size: %d\n\r", N);

        // Perform FFT
		arm_cfft_f32(&fft_instance, input, 0, 1);

        // Fill stack with a known pattern
        fill_stack_pattern_to_sp();

        // Measure stack before FFT
        register uint32_t *sp_before;
        __asm volatile ("mov %0, sp" : "=r" (sp_before));

        // Measure cycles before FFT
        uint32_t start_cycles = read_cycle_counter();

        // Compute magnitudes
        arm_cmplx_mag_f32(input, output, N);

        // Measure cycles after FFT
        uint32_t end_cycles = read_cycle_counter();

        // Measure stack usage
        uint32_t stack_used = measure_stack_usage();

        // Calculate cycle count
        uint32_t cycle_count = end_cycles - start_cycles;

    	// Print the frequency bins and magnitudes
    	printf("Frequency Bins and Magnitudes:\n\r");
    	for (int i = 0; i < N; i++) {
    		double frequency_resolution = (double)SAMPLING_FREQ / N;
    		double frequency = i * frequency_resolution;
    		printf(" %.2f , %.2f\n\r", frequency, output[i]);
    	}

        // Print results
        printf("  Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("  Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("\n");

        free(input);
        free(output);
    }

    return 0;
}
