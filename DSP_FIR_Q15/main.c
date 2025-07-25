#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <string.h> // For memcpy
#include <stdlib.h>


#define TEST_LENGTH_SAMPLES  320
#define SNR_THRESHOLD_F32    75.0f
#define BLOCK_SIZE           32
#define NUM_TAPS             8

/* ----------------------------------------------------------------------
** Q15 Input Signal & FIR Coefficients (Time-Reversed for CMSIS-DSP)
** ------------------------------------------------------------------- */
static q15_t signal_q15_in[TEST_LENGTH_SAMPLES] = {
    0, 16436, 19259, 12778, 12465, 22935, 31161, 24333, 7704, -2261, -0, 2261,
    -7704, -24333, -31161, -22936, -12465, -12778, -19259, -16436, -0, 16436,
    19259, 12778, 12465, 22935, 31161, 24333, 7704, -2261, -0, 2261, -7704,
    -24333, -31161, -22936, -12465, -12778, -19259, -16436, -0, 16436, 19259,
    12778, 12465, 22935, 31161, 24333, 7704, -2261, -0, 2261, -7704, -24333,
    -31161, -22935, -12465, -12778, -19259, -16436, -0, 16436, 19259, 12778,
    12465, 22936, 31161, 24333, 7704, -2261, -0, 2261, -7704, -24333, -31161,
    -22935, -12465, -12778, -19259, -16436, -0, 16436, 19259, 12778, 12465,
    22936, 31161, 24333, 7704, -2261, -0, 2261, -7704, -24333, -31161, -22936,
    -12465, -12778, -19259, -16436, -0, 16436, 19259, 12778, 12465, 22936,
    31161, 24333, 7704, -2261, 0, 2261, -7704, -24333, -31161, -22936, -12465,
    -12778, -19259, -16436, -0, 16436, 19259, 12778, 12465, 22936, 31161,
    24333, 7704, -2261, 0, 2261, -7704, -24333, -31161, -22936, -12465, -12778,
    -19259, -16436, -0, 16436, 19259, 12778, 12465, 22936, 31161, 24333, 7704,
    -2261, 0, 2261, -7704, -24333, -31161, -22936, -12465, -12778, -19259,
    -16436, -0, 16436, 19259, 12778, 12465, 22935, 31161, 24333, 7704, -2261,
    0, 2261, -7704, -24333, -31161, -22936, -12465, -12778, -19259, -16436,
    -0, 16436, 19259, 12778, 12465, 22935, 31161, 24333, 7704, -2261, -0, 2261,
    -7704, -24333, -31161, -22936, -12465, -12778, -19259, -16436, -0, 16436,
    19259, 12778, 12465, 22935, 31161, 24333, 7704, -2261, 0, 2261, -7704,
    -24333, -31161, -22936, -12465, -12778, -19259, -16436, 0, 16436, 19259,
    12778, 12465, 22935, 31161, 24333, 7704, -2261, -0, 2261, -7704, -24333,
    -31161, -22936, -12465, -12778, -19259, -16436, -0, 16436, 19259, 12778,
    12465, 22935, 31161, 24333, 7704, -2261, 0, 2261, -7704, -24333, -31161,
    -22936, -12465, -12778, -19259, -16436, 0, 16436, 19259, 12778, 12465,
    22935, 31161, 24333, 7704, -2261, -0, 2261, -7704, -24333, -31161, -22936,
    -12465, -12778, -19259, -16436, -0,
};

static q15_t firCoeffs[NUM_TAPS] = {
		2411, 4172, 5626, 6446, 6446, 5626, 4172, 2411 // Coefficients padded to make NUM_TAPS even
};
/* ----------------------------------------------------------------------
** Declare State Buffer and Output Buffer
** ------------------------------------------------------------------- */
static q15_t testOutputQ15[TEST_LENGTH_SAMPLES];
static q15_t firStateQ15[BLOCK_SIZE + NUM_TAPS - 1];

/* ----------------------------------------------------------------------
** Declare a buffer to store the converted Q15 output in floating-point
** ------------------------------------------------------------------- */
//static float32_t convertedOutput_f32[TEST_LENGTH_SAMPLES];

/* ----------------------------------------------------------------------
** FIR instance for Q15
** ------------------------------------------------------------------- */
arm_fir_instance_q15 S;


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

/*******************************************************************************
* Function Name: main
********************************************************************************
* Summary:
* This is the main function. It sets up a timer to trigger a periodic interrupt.
* The main while loop checks for the status of a flag set by the interrupt and
* toggles an LED at 1Hz to create an LED blinky. Will be achieving the 1Hz Blink
* rate based on the The LED_BLINK_TIMER_CLOCK_HZ and LED_BLINK_TIMER_PERIOD
* Macros,i.e. (LED_BLINK_TIMER_PERIOD + 1) / LED_BLINK_TIMER_CLOCK_HZ = X ,Here,
* X denotes the desired blink rate. The while loop also checks whether the
* 'Enter' key was pressed and stops/restarts LED blinking.
*
* Parameters:
*  none
*
* Return:
*  int
*
*******************************************************************************/
int main(void)
{
    cy_rslt_t result;

#if defined (CY_DEVICE_SECURE)
    cyhal_wdt_t wdt_obj;

    /* Clear watchdog timer so that it doesn't trigger a reset */
    result = cyhal_wdt_init(&wdt_obj, cyhal_wdt_get_max_timeout_ms());
    CY_ASSERT(CY_RSLT_SUCCESS == result);
    cyhal_wdt_free(&wdt_obj);
#endif /* #if defined (CY_DEVICE_SECURE) */

    /* Initialize the device and board peripherals */
    result = cybsp_init();

    /* Board init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }

    /* Enable global interrupts */
    __enable_irq();

    /* Initialize retarget-io to use the debug UART port */
    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
            CYBSP_DEBUG_UART_CTS,CYBSP_DEBUG_UART_RTS,CY_RETARGET_IO_BAUDRATE);

    /* retarget-io init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }

    /* Initialize the User LED */
    result = cyhal_gpio_init(CYBSP_USER_LED, CYHAL_GPIO_DIR_OUTPUT,
                             CYHAL_GPIO_DRIVE_STRONG, CYBSP_LED_STATE_OFF);

    /* GPIO init failed. Stop program execution */
    if (result != CY_RSLT_SUCCESS)
    {
        CY_ASSERT(0);
    }
    printf("****************** \n");
	printf("FIR Program for CMSIS DSP Library\n");
	uint32_t i;

	/* Call FIR init function to initialize the instance structure. */
	arm_fir_init_q15(&S, NUM_TAPS, firCoeffs, firStateQ15, BLOCK_SIZE);

	// Enable cycle counter
	enable_cycle_counter();

	// Start cycle count
	uint32_t start_cycles = read_cycle_counter();

	for (i = 0; i < TEST_LENGTH_SAMPLES / BLOCK_SIZE; i++) {
			arm_fir_q15(&S, signal_q15_in + (i * BLOCK_SIZE), testOutputQ15 + (i * BLOCK_SIZE), BLOCK_SIZE);
		}

	// End cycle count
	unsigned int end_cycles = read_cycle_counter();

	// Calculate the total cycle count
	unsigned int cycle_count = end_cycles - start_cycles;

	//convert_and_recover_q15_to_f32(testOutputQ15, convertedOutput_f32, TEST_LENGTH_SAMPLES);

	printf("\nFIR Output vs. Reference Output:\n\r");
	for (i = 0; i < TEST_LENGTH_SAMPLES; i++) {
		printf("Index %3lu: FIR Output (Q15) = %d\n\r", i, testOutputQ15[i]);
	}

	printf("Cycle Count for D25F: %u\n", cycle_count);

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
