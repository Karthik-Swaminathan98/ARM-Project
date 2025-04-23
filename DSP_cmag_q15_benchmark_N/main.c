#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <string.h> // For memcpy
#include <stdlib.h>

#define SINE_FREQ 50           // Frequency of the sine wave
#define SAMPLING_FREQ 256      // Constant sampling frequency
#define Q15_SCALE 32768        // Scaling factor for Q15 format
#define NUM_EXECUTIONS 10      // Number of executions for each FFT size
#define FFT_SIZES_COUNT 6      // Total number of FFT sizes to test

// Array of FFT sizes to test
const int FFT_SIZES[FFT_SIZES_COUNT] = {1024, 512, 256, 128, 64, 32};

void generate_sine_wave_q15(q15_t* input, int N, float signal_freq, float sampling_freq) {
    for (int i = 0; i < N; i++) {
        // Generate sine wave in float format
        float32_t value = sinf(2 * M_PI * signal_freq * i / sampling_freq);
        // Convert to Q15 format
        input[2 * i] = (q15_t)(value * Q15_SCALE); // Real part
        input[2 * i + 1] = 0;                     // Imaginary part (set to 0)
    }
}


void enable_cycle_counter() {
    // Enable DWT
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    // Reset cycle counter
    DWT->CYCCNT = 0;
    // Enable cycle counter
    DWT->CTRL |= DWT_CTRL_CYCCNTENA_Msk;
}

uint32_t read_cycle_counter() {
    return DWT->CYCCNT;
}

/*******************************************************************************
* Macros
*******************************************************************************/
/* LED blink timer clock value in Hz  */
#define LED_BLINK_TIMER_CLOCK_HZ          (10000)

/* LED blink timer period value */
#define LED_BLINK_TIMER_PERIOD            (9999)


/*******************************************************************************
* Global Variables
*******************************************************************************/
bool timer_interrupt_flag = false;
bool led_blink_active_flag = true;

/* Variable for storing character read from terminal */
uint8_t uart_read_value;

/* Timer object used for blinking the LED */
cyhal_timer_t led_blink_timer;


/*******************************************************************************
* Function Prototypes
*******************************************************************************/
void timer_init(void);
static void isr_timer(void *callback_arg, cyhal_timer_event_t event);

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
    printf("Starting FFT Benchmark Program\n");

    // Enable cycle counter
    enable_cycle_counter();

    for (int size_idx = 0; size_idx < FFT_SIZES_COUNT; size_idx++) {
        int N = FFT_SIZES[size_idx];

        // Allocate memory
        q15_t* input = (q15_t*)malloc(2 * N * sizeof(q15_t));
        q15_t* original_input = (q15_t*)malloc(2 * N * sizeof(q15_t));
        q15_t* magnitude = (q15_t*)malloc(N * sizeof(q15_t));
        q15_t* magnitude_reference = (q15_t*)malloc(N * sizeof(q15_t)); // To store the reference magnitudes
        unsigned int cycle_counts[NUM_EXECUTIONS]; // Cycle counts for each execution

        if (input == NULL || original_input == NULL || magnitude == NULL || magnitude_reference == NULL) {
            printf("Memory allocation failed for FFT size N = %d\n", N);
            return -1;
        }

        // Generate 50 Hz sine wave
        generate_sine_wave_q15(original_input, N, SINE_FREQ, SAMPLING_FREQ);

        // Create FFT instance
        arm_cfft_instance_q15 fft_instance;
        if (arm_cfft_init_q15(&fft_instance, N) != ARM_MATH_SUCCESS) {
            printf("FFT initialization failed for N = %d\n", N);
            return -1;
        }

        printf("FFT Size: %d\n", N);
        printf("Input Data,Frequency (Hz)");
        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            printf(",Execution %d Output", execution + 1);
        }
        printf("\n");

        // Variable to track consistency
        int magnitudes_consistent = 1; // Assume consistent until proven otherwise

        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            memcpy(input, original_input, 2 * N * sizeof(q15_t));

            arm_cfft_q15(&fft_instance, input, 0, 1); // FFT

            // Reset and start cycle counter
            uint32_t start_cycles = read_cycle_counter();

            arm_cmplx_mag_q15(input, magnitude, N); // Compute magnitudes

            uint32_t end_cycles = read_cycle_counter();
            cycle_counts[execution] = end_cycles - start_cycles;

            // Store magnitudes from the first execution as the reference
            if (execution == 0) {
                memcpy(magnitude_reference, magnitude, N * sizeof(q15_t));
            } else {
                // Compare current magnitudes to the reference
                for (int i = 0; i < N; i++) {
                    if (abs(magnitude[i] - magnitude_reference[i]) > 1) { // Allow a tolerance of 1
                        magnitudes_consistent = 0; // Inconsistent
                        break;
                    }
                }
            }
        }

        // Print frequency bins and magnitudes
//        for (int i = 0; i < N; i++) {
//            float frequency_bin = (i * ((float32_t)SAMPLING_FREQ / (float32_t)N));
//            //float normalized_magnitude = magnitude[i] / (float32_t)N;
//
//            printf("%.4f,%.2f", original_input[2 * i] / (float32_t)Q15_SCALE, frequency_bin);
//
//            for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
//                printf(",%.4f", magnitude[i] / (float32_t)Q15_SCALE);
//            }
//            printf("\n");
//        }

        // Average cycle count
        unsigned int total_cycle_count = 0;
        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            total_cycle_count += cycle_counts[execution];
        }
        unsigned int average_cycle_count = total_cycle_count / NUM_EXECUTIONS;

        // Print cycle counts
        printf("Cycle Count,");
        printf("Average: %u", average_cycle_count);
        for (int execution = 0; execution < NUM_EXECUTIONS; execution++) {
            printf(",%u", cycle_counts[execution]);
        }
        printf("\n");

        // Print consistency results
        if (magnitudes_consistent) {
            printf("All magnitudes are consistent across all executions for FFT size N = %d\n", N);
        } else {
            printf("Inconsistent magnitudes detected for FFT size N = %d\n", N);
        }

        // Free allocated memory
        free(input);
        free(original_input);
        free(magnitude);
        free(magnitude_reference);
    }

    return 0;
}



/*******************************************************************************
* Function Name: timer_init
********************************************************************************
* Summary:
* This function creates and configures a Timer object. The timer ticks
* continuously and produces a periodic interrupt on every terminal count
* event. The period is defined by the 'period' and 'compare_value' of the
* timer configuration structure 'led_blink_timer_cfg'. Without any changes,
* this application is designed to produce an interrupt every 1 second.
*
* Parameters:
*  none
*
* Return :
*  void
*
*******************************************************************************/
 void timer_init(void)
 {
    cy_rslt_t result;

    const cyhal_timer_cfg_t led_blink_timer_cfg =
    {
        .compare_value = 0,                 /* Timer compare value, not used */
        .period = LED_BLINK_TIMER_PERIOD,   /* Defines the timer period */
        .direction = CYHAL_TIMER_DIR_UP,    /* Timer counts up */
        .is_compare = false,                /* Don't use compare mode */
        .is_continuous = true,              /* Run timer indefinitely */
        .value = 0                          /* Initial value of counter */
    };

    /* Initialize the timer object. Does not use input pin ('pin' is NC) and
     * does not use a pre-configured clock source ('clk' is NULL). */
    result = cyhal_timer_init(&led_blink_timer, NC, NULL);

    /* timer init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }

    /* Configure timer period and operation mode such as count direction,
       duration */
    cyhal_timer_configure(&led_blink_timer, &led_blink_timer_cfg);

    /* Set the frequency of timer's clock source */
    cyhal_timer_set_frequency(&led_blink_timer, LED_BLINK_TIMER_CLOCK_HZ);

    /* Assign the ISR to execute on timer interrupt */
    cyhal_timer_register_callback(&led_blink_timer, isr_timer, NULL);

    /* Set the event on which timer interrupt occurs and enable it */
    cyhal_timer_enable_event(&led_blink_timer, CYHAL_TIMER_IRQ_TERMINAL_COUNT,
                              7, true);

    /* Start the timer with the configured settings */
    cyhal_timer_start(&led_blink_timer);
 }


/*******************************************************************************
* Function Name: isr_timer
********************************************************************************
* Summary:
* This is the interrupt handler function for the timer interrupt.
*
* Parameters:
*    callback_arg    Arguments passed to the interrupt callback
*    event            Timer/counter interrupt triggers
*
* Return:
*  void
*******************************************************************************/
static void isr_timer(void *callback_arg, cyhal_timer_event_t event)
{
    (void) callback_arg;
    (void) event;

    /* Set the interrupt flag and process it from the main while(1) loop */
    timer_interrupt_flag = true;
}

/* [] END OF FILE */
