#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define SINE_FREQ 50           // Frequency of the sine wave
#define SAMPLING_FREQ 256      // Constant sampling frequency
#define NUM_EXECUTIONS 10      // Number of executions for each FFT size
#define FFT_SIZES_COUNT 6      // Total number of FFT sizes to test

// Array of FFT sizes to test
const int FFT_SIZES[FFT_SIZES_COUNT] = {1024, 512, 256, 128, 64, 32};

void generate_sine_wave_f32(float32_t* input, int N, float signal_freq, float sampling_freq) {
    for (int i = 0; i < N; i++) {
        float32_t value = sinf(2 * M_PI * signal_freq * i / sampling_freq);
        input[2 * i] = value;  // Real part
        input[2 * i + 1] = 0;  // Imaginary part (set to 0)
    }
}

void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;  // Enable DWT
    DWT->CYCCNT = 0;                                // Reset cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;            // Enable cycle counter
}

uint32_t read_cycle_counter() {
    return DWT->CYCCNT;
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

    printf("****************** \n");
    printf("Starting FFT Benchmark Program (f32)\n");

    enable_cycle_counter();

    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        // Allocate memory
        float32_t* input = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* original_input = (float32_t*)malloc(2 * N * sizeof(float32_t));
        float32_t* magnitude = (float32_t*)malloc(N * sizeof(float32_t));
        float32_t* magnitude_reference = (float32_t*)malloc(N * sizeof(float32_t));
        unsigned int cycle_counts[NUM_EXECUTIONS];

        if (input == NULL || original_input == NULL || magnitude == NULL || magnitude_reference == NULL) {
            printf("Memory allocation failed for FFT size N = %d\n", N);
            return -1;
        }

        // Generate 50 Hz sine wave
        generate_sine_wave_f32(original_input, N, SINE_FREQ, SAMPLING_FREQ);

        // Create FFT instance
        arm_cfft_instance_f32 fft_instance;
        if (arm_cfft_init_f32(&fft_instance, N) != ARM_MATH_SUCCESS) {
            printf("FFT initialization failed for N = %d\n", N);
            return -1;
        }

        printf("FFT Size: %d\n", N);
        printf("Input Data,Frequency (Hz)");
        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            printf(",Execution %d Output", execution + 1);
        }
        printf("\n");

        int magnitudes_consistent = 1;

        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            memcpy(input, original_input, 2 * N * sizeof(float32_t));

            arm_cfft_f32(&fft_instance, input, 0, 1);

            uint32_t start_cycles = read_cycle_counter();

            arm_cmplx_mag_f32(input, magnitude, N);

            uint32_t end_cycles = read_cycle_counter();
            cycle_counts[execution] = end_cycles - start_cycles;

            if (execution == 0) {
                memcpy(magnitude_reference, magnitude, N * sizeof(float32_t));
            } else {
                for (int i = 0; i < N; i++) {
                    if (fabs(magnitude[i] - magnitude_reference[i]) > 0.001) {
                        magnitudes_consistent = 0;
                        break;
                    }
                }
            }
        }

//        for (int i = 0; i < N; i++) {
//            float frequency_bin = (i * ((float32_t)SAMPLING_FREQ / (float32_t)N));
//            printf("%.4f,%.2f", original_input[2 * i], frequency_bin);
//            for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
//                printf(",%.4f", magnitude[i]);
//            }
//            printf("\n");
//        }

        unsigned int total_cycle_count = 0;
        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            total_cycle_count += cycle_counts[execution];
        }
        unsigned int average_cycle_count = total_cycle_count / NUM_EXECUTIONS;

        printf("Cycle Count,Average: %u", average_cycle_count);
        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            printf(",%u", cycle_counts[execution]);
        }
        printf("\n");

        if (magnitudes_consistent) {
            printf("All magnitudes are consistent across all executions for FFT size N = %d\n", N);
        } else {
            printf("Inconsistent magnitudes detected for FFT size N = %d\n", N);
        }

        free(input);
        free(original_input);
        free(magnitude);
        free(magnitude_reference);
    }

    return 0;
}
