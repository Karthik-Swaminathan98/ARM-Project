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

void generate_sine_wave_q15(q15_t* input, int N, float signal_freq, float sampling_freq) {
    for (int i = 0; i < N; i++) {
        float value = sinf(2 * M_PI * signal_freq * i / sampling_freq);
        input[2 * i] = (q15_t)(value * 32767.0f);  // Real part (scaled to Q15 format)
        input[2 * i + 1] = 0;                     // Imaginary part (set to 0)
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
    printf("Starting FFT Stack Usage Benchmark Program (Q15)\n\r");

    enable_cycle_counter();

    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        // Allocate memory
        q15_t* input = (q15_t*)malloc(2 * N * sizeof(q15_t));
        q15_t* output = (q15_t*)malloc(2 * N * sizeof(q15_t)); // Output buffer for magnitudes

        if (input == NULL || output == NULL) {
            printf("Memory allocation failed for FFT size N = %d\n\r", N);
            return -1;
        }

        // Generate 50 Hz sine wave
        generate_sine_wave_q15(input, N, SINE_FREQ, SAMPLING_FREQ);

        // Create FFT instance
        arm_cfft_instance_q15 fft_instance;
        if (arm_cfft_init_q15(&fft_instance, N) != ARM_MATH_SUCCESS) {
            printf("FFT initialization failed for N = %d\n\r", N);
            free(input);
            free(output);
            continue;
        }

        printf("FFT Size: %d\n\r", N);

        // Fill stack with a known pattern
        fill_stack_pattern_to_sp();

        // Measure stack before FFT
        register uint32_t *sp_before;
        __asm volatile ("mov %0, sp" : "=r" (sp_before));

        // Measure cycles before FFT
        uint32_t start_cycles = read_cycle_counter();

        // Perform FFT
        arm_cfft_q15(&fft_instance, input, 0, 1);

        // Measure cycles after FFT
        uint32_t end_cycles = read_cycle_counter();

        // Measure stack usage
        uint32_t stack_used = measure_stack_usage();

        // Calculate cycle count
        uint32_t cycle_count = end_cycles - start_cycles;

        // Compute magnitudes
        arm_cmplx_mag_q15(input, output, N);

        // Print results
        printf("  Stack Used: %lu bytes\n\r", (unsigned long)stack_used);
        printf("  Cycle Count: %lu\n\r", (unsigned long)cycle_count);
        printf("\n");

        free(input);
        free(output);
    }

    return 0;
}
