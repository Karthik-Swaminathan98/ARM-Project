#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>

#define SINE_FREQ 50          // Frequency of the sine wave
#define SAMPLING_FREQ 256     // Sampling frequency
#define FFT_SIZE 128          // FFT size

// Allocate input and output arrays
float32_t input[FFT_SIZE * 2]; // Interleaved input (real + imag)
float32_t output[FFT_SIZE];    // Magnitude output
extern uint32_t __StackLimit;
extern uint32_t __StackTop;

void generate_sine_wave(float32_t* input, int N) {
    for (int i = 0; i < N; i++) {
        input[2 * i] = sin(2 * M_PI * SINE_FREQ * i / SAMPLING_FREQ); // Real part
        input[2 * i + 1] = 0.0; // Imaginary part
    }
}

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

void print_stack_info(void) {
    uint32_t stack_top = (uint32_t)&__StackTop;
    uint32_t stack_limit = (uint32_t)&__StackLimit;
    uint32_t stack_size = stack_top - stack_limit;

    printf("Stack Top    : 0x%08lX\n\r", (unsigned long)stack_top);
    printf("Stack Bottom : 0x%08lX\n\r", (unsigned long)stack_limit);
    printf("Stack Size   : %lu bytes\n\r", (unsigned long)stack_size);
}

int main(void) {
    cy_rslt_t result;

    /* Initialize the device and board peripherals */
    result = cybsp_init();

    /* Board init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    /* Enable global interrupts */
    __enable_irq();

    /* Initialize retarget-io to use the debug UART port */
    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                                    CYBSP_DEBUG_UART_CTS, CYBSP_DEBUG_UART_RTS, CY_RETARGET_IO_BAUDRATE);

    /* Retarget-io init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    printf("Starting FFT stack usage test...\n\r");

    // Enable cycle counter
    enable_cycle_counter();

    // Define the number of samples
    int N = FFT_SIZE;

    // Generate a sine wave signal
    generate_sine_wave(input, N);

    // Create the FFT instance
    arm_cfft_instance_f32 fft_instance;
    arm_cfft_init_f32(&fft_instance, N);

    // Fill stack with a known pattern
    fill_stack_pattern_to_sp();
    register uint32_t *sp_before;
    __asm volatile ("mov %0, sp" : "=r" (sp_before));

    // Measure cycles before FFT
    uint32_t start_cycles = read_cycle_counter();

    // Perform FFT
    arm_cfft_f32(&fft_instance, input, 0, 1);

    // Measure cycles after FFT
    uint32_t end_cycles = read_cycle_counter();

    // Determine stack usage
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp_before) {
        if (*p != 0xAAAAAAAA) {
            break;
        }
        p++;
    }

    uint32_t stack_used = ((uint32_t)sp_before - (uint32_t)p); // in bytes

    // Calculate cycle count
    uint32_t cycle_count = end_cycles - start_cycles;

    // Compute magnitudes
    arm_cmplx_mag_f32(input, output, N);

    // Print the frequency bins and magnitudes
    printf("Frequency Bins and Magnitudes:\n\r");
    for (int i = 0; i < N; i++) {
        double frequency_resolution = (double)SAMPLING_FREQ / N;
        double frequency = i * frequency_resolution;
        printf(" %.2f , %.2f\n\r", frequency, output[i]);
    }

    // Print stack and performance information
    print_stack_info();
    printf("Stack used by arm_cfft_f32: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu\n\r", (unsigned long)cycle_count);

    return 0;
}
