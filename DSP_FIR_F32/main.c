#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>

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

/* ----------------------------------------------------------------------
** Macro Defines
** ------------------------------------------------------------------- */

#define TEST_LENGTH_SAMPLES  320
/*

This SNR is a bit small. Need to understand why
this example is not giving better SNR ...

*/
#define SNR_THRESHOLD_F32    75.0f
#define BLOCK_SIZE            32

#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
/* Must be a multiple of 16 */
#define NUM_TAPS_ARRAY_SIZE              32
#else
#define NUM_TAPS_ARRAY_SIZE              29
#endif

#define NUM_TAPS              29

/* -------------------------------------------------------------------
 * The input signal and reference output (computed with MATLAB)
 * are defined externally in arm_fir_lpf_data.c.
 * ------------------------------------------------------------------- */

//extern float32_t testInput_f32_1kHz_15kHz[TEST_LENGTH_SAMPLES];
//extern float32_t refOutput[TEST_LENGTH_SAMPLES];

/* -------------------------------------------------------------------
 * Declare Test output buffer
 * ------------------------------------------------------------------- */

static float32_t testOutput[TEST_LENGTH_SAMPLES];

/* -------------------------------------------------------------------
 * Declare State buffer of size (numTaps + blockSize - 1)
 * ------------------------------------------------------------------- */
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
static float32_t firStateF32[2 * BLOCK_SIZE + NUM_TAPS - 1];
#else
static float32_t firStateF32[BLOCK_SIZE + NUM_TAPS - 1];
#endif

/* ----------------------------------------------------------------------
** FIR Coefficients buffer generated using fir1() MATLAB function.
** fir1(28, 6/24)
** ------------------------------------------------------------------- */
#if defined(ARM_MATH_MVEF) && !defined(ARM_MATH_AUTOVECTORIZE)
const float32_t firCoeffs32[NUM_TAPS_ARRAY_SIZE] = {
  -0.0018225230f, -0.0015879294f, +0.0000000000f, +0.0036977508f, +0.0080754303f, +0.0085302217f, -0.0000000000f, -0.0173976984f,
  -0.0341458607f, -0.0333591565f, +0.0000000000f, +0.0676308395f, +0.1522061835f, +0.2229246956f, +0.2504960933f, +0.2229246956f,
  +0.1522061835f, +0.0676308395f, +0.0000000000f, -0.0333591565f, -0.0341458607f, -0.0173976984f, -0.0000000000f, +0.0085302217f,
  +0.0080754303f, +0.0036977508f, +0.0000000000f, -0.0015879294f, -0.0018225230f, 0.0f,0.0f,0.0f
};
#else
const float32_t firCoeffs32[NUM_TAPS_ARRAY_SIZE] = {
  -0.0018225230f, -0.0015879294f, +0.0000000000f, +0.0036977508f, +0.0080754303f, +0.0085302217f, -0.0000000000f, -0.0173976984f,
  -0.0341458607f, -0.0333591565f, +0.0000000000f, +0.0676308395f, +0.1522061835f, +0.2229246956f, +0.2504960933f, +0.2229246956f,
  +0.1522061835f, +0.0676308395f, +0.0000000000f, -0.0333591565f, -0.0341458607f, -0.0173976984f, -0.0000000000f, +0.0085302217f,
  +0.0080754303f, +0.0036977508f, +0.0000000000f, -0.0015879294f, -0.0018225230f
};
#endif

/* ----------------------------------------------------------------------
** Test input signal contains 1000Hz + 15000 Hz
** ------------------------------------------------------------------- */

const float32_t testInput_f32_1kHz_15kHz[320] =
{
+0.0000000000f, +0.5924659585f, -0.0947343455f, +0.1913417162f, +1.0000000000f, +0.4174197128f, +0.3535533906f, +1.2552931065f,
+0.8660254038f, +0.4619397663f, +1.3194792169f, +1.1827865776f, +0.5000000000f, +1.1827865776f, +1.3194792169f, +0.4619397663f,
+0.8660254038f, +1.2552931065f, +0.3535533906f, +0.4174197128f, +1.0000000000f, +0.1913417162f, -0.0947343455f, +0.5924659585f,
-0.0000000000f, -0.5924659585f, +0.0947343455f, -0.1913417162f, -1.0000000000f, -0.4174197128f, -0.3535533906f, -1.2552931065f,
-0.8660254038f, -0.4619397663f, -1.3194792169f, -1.1827865776f, -0.5000000000f, -1.1827865776f, -1.3194792169f, -0.4619397663f,
-0.8660254038f, -1.2552931065f, -0.3535533906f, -0.4174197128f, -1.0000000000f, -0.1913417162f, +0.0947343455f, -0.5924659585f,
+0.0000000000f, +0.5924659585f, -0.0947343455f, +0.1913417162f, +1.0000000000f, +0.4174197128f, +0.3535533906f, +1.2552931065f,
+0.8660254038f, +0.4619397663f, +1.3194792169f, +1.1827865776f, +0.5000000000f, +1.1827865776f, +1.3194792169f, +0.4619397663f,
+0.8660254038f, +1.2552931065f, +0.3535533906f, +0.4174197128f, +1.0000000000f, +0.1913417162f, -0.0947343455f, +0.5924659585f,
+0.0000000000f, -0.5924659585f, +0.0947343455f, -0.1913417162f, -1.0000000000f, -0.4174197128f, -0.3535533906f, -1.2552931065f,
-0.8660254038f, -0.4619397663f, -1.3194792169f, -1.1827865776f, -0.5000000000f, -1.1827865776f, -1.3194792169f, -0.4619397663f,
-0.8660254038f, -1.2552931065f, -0.3535533906f, -0.4174197128f, -1.0000000000f, -0.1913417162f, +0.0947343455f, -0.5924659585f,
+0.0000000000f, +0.5924659585f, -0.0947343455f, +0.1913417162f, +1.0000000000f, +0.4174197128f, +0.3535533906f, +1.2552931065f,
+0.8660254038f, +0.4619397663f, +1.3194792169f, +1.1827865776f, +0.5000000000f, +1.1827865776f, +1.3194792169f, +0.4619397663f,
+0.8660254038f, +1.2552931065f, +0.3535533906f, +0.4174197128f, +1.0000000000f, +0.1913417162f, -0.0947343455f, +0.5924659585f,
+0.0000000000f, -0.5924659585f, +0.0947343455f, -0.1913417162f, -1.0000000000f, -0.4174197128f, -0.3535533906f, -1.2552931065f,
-0.8660254038f, -0.4619397663f, -1.3194792169f, -1.1827865776f, -0.5000000000f, -1.1827865776f, -1.3194792169f, -0.4619397663f,
-0.8660254038f, -1.2552931065f, -0.3535533906f, -0.4174197128f, -1.0000000000f, -0.1913417162f, +0.0947343455f, -0.5924659585f,
-0.0000000000f, +0.5924659585f, -0.0947343455f, +0.1913417162f, +1.0000000000f, +0.4174197128f, +0.3535533906f, +1.2552931065f,
+0.8660254038f, +0.4619397663f, +1.3194792169f, +1.1827865776f, +0.5000000000f, +1.1827865776f, +1.3194792169f, +0.4619397663f,
+0.8660254038f, +1.2552931065f, +0.3535533906f, +0.4174197128f, +1.0000000000f, +0.1913417162f, -0.0947343455f, +0.5924659585f,
-0.0000000000f, -0.5924659585f, +0.0947343455f, -0.1913417162f, -1.0000000000f, -0.4174197128f, -0.3535533906f, -1.2552931065f,
-0.8660254038f, -0.4619397663f, -1.3194792169f, -1.1827865776f, -0.5000000000f, -1.1827865776f, -1.3194792169f, -0.4619397663f,
-0.8660254038f, -1.2552931065f, -0.3535533906f, -0.4174197128f, -1.0000000000f, -0.1913417162f, +0.0947343455f, -0.5924659585f,
+0.0000000000f, +0.5924659585f, -0.0947343455f, +0.1913417162f, +1.0000000000f, +0.4174197128f, +0.3535533906f, +1.2552931065f,
+0.8660254038f, +0.4619397663f, +1.3194792169f, +1.1827865776f, +0.5000000000f, +1.1827865776f, +1.3194792169f, +0.4619397663f,
+0.8660254038f, +1.2552931065f, +0.3535533906f, +0.4174197128f, +1.0000000000f, +0.1913417162f, -0.0947343455f, +0.5924659585f,
+0.0000000000f, -0.5924659585f, +0.0947343455f, -0.1913417162f, -1.0000000000f, -0.4174197128f, -0.3535533906f, -1.2552931065f,
-0.8660254038f, -0.4619397663f, -1.3194792169f, -1.1827865776f, -0.5000000000f, -1.1827865776f, -1.3194792169f, -0.4619397663f,
-0.8660254038f, -1.2552931065f, -0.3535533906f, -0.4174197128f, -1.0000000000f, -0.1913417162f, +0.0947343455f, -0.5924659585f,
-0.0000000000f, +0.5924659585f, -0.0947343455f, +0.1913417162f, +1.0000000000f, +0.4174197128f, +0.3535533906f, +1.2552931065f,
+0.8660254038f, +0.4619397663f, +1.3194792169f, +1.1827865776f, +0.5000000000f, +1.1827865776f, +1.3194792169f, +0.4619397663f,
+0.8660254038f, +1.2552931065f, +0.3535533906f, +0.4174197128f, +1.0000000000f, +0.1913417162f, -0.0947343455f, +0.5924659585f,
+0.0000000000f, -0.5924659585f, +0.0947343455f, -0.1913417162f, -1.0000000000f, -0.4174197128f, -0.3535533906f, -1.2552931065f,
-0.8660254038f, -0.4619397663f, -1.3194792169f, -1.1827865776f, -0.5000000000f, -1.1827865776f, -1.3194792169f, -0.4619397663f,
-0.8660254038f, -1.2552931065f, -0.3535533906f, -0.4174197128f, -1.0000000000f, -0.1913417162f, +0.0947343455f, -0.5924659585f,
-0.0000000000f, +0.5924659585f, -0.0947343455f, +0.1913417162f, +1.0000000000f, +0.4174197128f, +0.3535533906f, +1.2552931065f,
+0.8660254038f, +0.4619397663f, +1.3194792169f, +1.1827865776f, +0.5000000000f, +1.1827865776f, +1.3194792169f, +0.4619397663f,
+0.8660254038f, +1.2552931065f, +0.3535533906f, +0.4174197128f, +1.0000000000f, +0.1913417162f, -0.0947343455f, +0.5924659585f,
+0.0000000000f, -0.5924659585f, +0.0947343455f, -0.1913417162f, -1.0000000000f, -0.4174197128f, -0.3535533906f, -1.2552931065f,
};

const float32_t refOutput[320] =
{
+0.0000000000f, -0.0010797829f, -0.0007681386f, -0.0001982932f, +0.0000644313f, +0.0020854271f, +0.0036891871f, +0.0015855941f,
-0.0026280805f, -0.0075907658f, -0.0119390538f, -0.0086665968f, +0.0088981202f, +0.0430539279f, +0.0974468742f, +0.1740405600f,
+0.2681416601f, +0.3747720089f, +0.4893362230f, +0.6024154672f, +0.7058740791f, +0.7968348987f, +0.8715901940f, +0.9277881093f,
+0.9682182661f, +0.9934674267f, +1.0012052245f, +0.9925859371f, +0.9681538347f, +0.9257026822f, +0.8679010068f, +0.7952493046f,
+0.7085021596f, +0.6100062330f, +0.5012752767f, +0.3834386057f, +0.2592435399f, +0.1309866321f, -0.0000000000f, -0.1309866321f,
-0.2592435399f, -0.3834386057f, -0.5012752767f, -0.6100062330f, -0.7085021596f, -0.7952493046f, -0.8679010068f, -0.9257026822f,
-0.9681538347f, -0.9936657199f, -1.0019733630f, -0.9936657199f, -0.9681538347f, -0.9257026822f, -0.8679010068f, -0.7952493046f,
-0.7085021596f, -0.6100062330f, -0.5012752767f, -0.3834386057f, -0.2592435399f, -0.1309866321f, +0.0000000000f, +0.1309866321f,
+0.2592435399f, +0.3834386057f, +0.5012752767f, +0.6100062330f, +0.7085021596f, +0.7952493046f, +0.8679010068f, +0.9257026822f,
+0.9681538347f, +0.9936657199f, +1.0019733630f, +0.9936657199f, +0.9681538347f, +0.9257026822f, +0.8679010068f, +0.7952493046f,
+0.7085021596f, +0.6100062330f, +0.5012752767f, +0.3834386057f, +0.2592435399f, +0.1309866321f, -0.0000000000f, -0.1309866321f,
-0.2592435399f, -0.3834386057f, -0.5012752767f, -0.6100062330f, -0.7085021596f, -0.7952493046f, -0.8679010068f, -0.9257026822f,
-0.9681538347f, -0.9936657199f, -1.0019733630f, -0.9936657199f, -0.9681538347f, -0.9257026822f, -0.8679010068f, -0.7952493046f,
-0.7085021596f, -0.6100062330f, -0.5012752767f, -0.3834386057f, -0.2592435399f, -0.1309866321f, +0.0000000000f, +0.1309866321f,
+0.2592435399f, +0.3834386057f, +0.5012752767f, +0.6100062330f, +0.7085021596f, +0.7952493046f, +0.8679010068f, +0.9257026822f,
+0.9681538347f, +0.9936657199f, +1.0019733630f, +0.9936657199f, +0.9681538347f, +0.9257026822f, +0.8679010068f, +0.7952493046f,
+0.7085021596f, +0.6100062330f, +0.5012752767f, +0.3834386057f, +0.2592435399f, +0.1309866321f, -0.0000000000f, -0.1309866321f,
-0.2592435399f, -0.3834386057f, -0.5012752767f, -0.6100062330f, -0.7085021596f, -0.7952493046f, -0.8679010068f, -0.9257026822f,
-0.9681538347f, -0.9936657199f, -1.0019733630f, -0.9936657199f, -0.9681538347f, -0.9257026822f, -0.8679010068f, -0.7952493046f,
-0.7085021596f, -0.6100062330f, -0.5012752767f, -0.3834386057f, -0.2592435399f, -0.1309866321f, +0.0000000000f, +0.1309866321f,
+0.2592435399f, +0.3834386057f, +0.5012752767f, +0.6100062330f, +0.7085021596f, +0.7952493046f, +0.8679010068f, +0.9257026822f,
+0.9681538347f, +0.9936657199f, +1.0019733630f, +0.9936657199f, +0.9681538347f, +0.9257026822f, +0.8679010068f, +0.7952493046f,
+0.7085021596f, +0.6100062330f, +0.5012752767f, +0.3834386057f, +0.2592435399f, +0.1309866321f, +0.0000000000f, -0.1309866321f,
-0.2592435399f, -0.3834386057f, -0.5012752767f, -0.6100062330f, -0.7085021596f, -0.7952493046f, -0.8679010068f, -0.9257026822f,
-0.9681538347f, -0.9936657199f, -1.0019733630f, -0.9936657199f, -0.9681538347f, -0.9257026822f, -0.8679010068f, -0.7952493046f,
-0.7085021596f, -0.6100062330f, -0.5012752767f, -0.3834386057f, -0.2592435399f, -0.1309866321f, +0.0000000000f, +0.1309866321f,
+0.2592435399f, +0.3834386057f, +0.5012752767f, +0.6100062330f, +0.7085021596f, +0.7952493046f, +0.8679010068f, +0.9257026822f,
+0.9681538347f, +0.9936657199f, +1.0019733630f, +0.9936657199f, +0.9681538347f, +0.9257026822f, +0.8679010068f, +0.7952493046f,
+0.7085021596f, +0.6100062330f, +0.5012752767f, +0.3834386057f, +0.2592435399f, +0.1309866321f, +0.0000000000f, -0.1309866321f,
-0.2592435399f, -0.3834386057f, -0.5012752767f, -0.6100062330f, -0.7085021596f, -0.7952493046f, -0.8679010068f, -0.9257026822f,
-0.9681538347f, -0.9936657199f, -1.0019733630f, -0.9936657199f, -0.9681538347f, -0.9257026822f, -0.8679010068f, -0.7952493046f,
-0.7085021596f, -0.6100062330f, -0.5012752767f, -0.3834386057f, -0.2592435399f, -0.1309866321f, -0.0000000000f, +0.1309866321f,
+0.2592435399f, +0.3834386057f, +0.5012752767f, +0.6100062330f, +0.7085021596f, +0.7952493046f, +0.8679010068f, +0.9257026822f,
+0.9681538347f, +0.9936657199f, +1.0019733630f, +0.9936657199f, +0.9681538347f, +0.9257026822f, +0.8679010068f, +0.7952493046f,
+0.7085021596f, +0.6100062330f, +0.5012752767f, +0.3834386057f, +0.2592435399f, +0.1309866321f, +0.0000000000f, -0.1309866321f,
-0.2592435399f, -0.3834386057f, -0.5012752767f, -0.6100062330f, -0.7085021596f, -0.7952493046f, -0.8679010068f, -0.9257026822f,
-0.9681538347f, -0.9936657199f, -1.0019733630f, -0.9936657199f, -0.9681538347f, -0.9257026822f, -0.8679010068f, -0.7952493046f,
-0.7085021596f, -0.6100062330f, -0.5012752767f, -0.3834386057f, -0.2592435399f, -0.1309866321f, +0.0000000000f, +0.1309866321f,
+0.2592435399f, +0.3834386057f, +0.5012752767f, +0.6100062330f, +0.7085021596f, +0.7952493046f, +0.8679010068f, +0.9257026822f,
+0.9681538347f, +0.9936657199f, +1.0019733630f, +0.9936657199f, +0.9681538347f, +0.9257026822f, +0.8679010068f, +0.7952493046f
};
/* ------------------------------------------------------------------
 * Global variables for FIR LPF Example
 * ------------------------------------------------------------------- */

uint32_t blockSize = BLOCK_SIZE;
uint32_t numBlocks = TEST_LENGTH_SAMPLES/BLOCK_SIZE;

float32_t  snr;
float arm_snr_f32(const float *pRef, float *pTest,  uint32_t buffSize);
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
	// Enable cycle counter
	enable_cycle_counter();

    printf("FIR Program Start\n\r");
    uint32_t i;
    arm_fir_instance_f32 S;
    arm_status status;
    const float32_t *inputF32;
    float32_t *outputF32;

  /* Initialize input and output buffer pointers */
    inputF32 = &testInput_f32_1kHz_15kHz[0];
    outputF32 = &testOutput[0];

  /* Call FIR init function to initialize the instance structure. */
    arm_fir_init_f32(&S, NUM_TAPS, (float32_t *)&firCoeffs32[0], &firStateF32[0], blockSize);

  /* ----------------------------------------------------------------------
  ** Call the FIR process function for every blockSize samples
  ** ------------------------------------------------------------------- */
    // Start cycle count
    uint32_t start_cycles = read_cycle_counter();

    for(i=0; i < numBlocks; i++)
    {
    	arm_fir_f32(&S, inputF32 + (i * blockSize), outputF32 + (i * blockSize), blockSize);
    }

    // End cycle count
	uint32_t end_cycles = read_cycle_counter();

	// Calculate and print cycle count
	uint32_t cycle_count = end_cycles - start_cycles;

    /* ----------------------------------------------------------------------
    ** Compare the generated output against the reference output computed
    ** in MATLAB.
    ** ------------------------------------------------------------------- */
	printf("\nFIR Output vs. Reference Output:\n");
	for (uint32_t i = 0; i < TEST_LENGTH_SAMPLES; i++) {
		printf("Index %3lu: FIR Output = %f, Reference Output = %f\n\r", i, testOutput[i], refOutput[i]);
	}

    snr = arm_snr_f32(&refOutput[0], &testOutput[0], TEST_LENGTH_SAMPLES);

    status = (snr < SNR_THRESHOLD_F32) ? ARM_MATH_TEST_FAILURE : ARM_MATH_SUCCESS;

    if (status != ARM_MATH_SUCCESS)
	{
		printf("FAILURE: SNR = %f (Threshold = %f)\n", snr, SNR_THRESHOLD_F32);
	}
	else
	{
		printf("SUCCESS: SNR = %f (Threshold = %f)\n", snr, SNR_THRESHOLD_F32);
	}

    printf("Cycle Count: %lu\n", (unsigned long)cycle_count);
    return 0;
}

/**
 * @brief  Caluclation of SNR
 * @param[in]  pRef 	Pointer to the reference buffer
 * @param[in]  pTest	Pointer to the test buffer
 * @param[in]  buffSize	total number of samples
 * @return     SNR
 * The function Caluclates signal to noise ratio for the reference output
 * and test output
 */

float arm_snr_f32(const float *pRef, float *pTest, uint32_t buffSize)
{
  float EnergySignal = 0.0, EnergyError = 0.0;
  uint32_t i;
  float SNR;
  int temp;
  int *test;

  for (i = 0; i < buffSize; i++)
    {
 	  /* Checking for a NAN value in pRef array */
	  test =   (int *)(&pRef[i]);
      temp =  *test;

	  if (temp == 0x7FC00000)
	  {
	  		return(0);
	  }

	  /* Checking for a NAN value in pTest array */
	  test =   (int *)(&pTest[i]);
      temp =  *test;

	  if (temp == 0x7FC00000)
	  {
	  		return(0);
	  }
      EnergySignal += pRef[i] * pRef[i];
      EnergyError += (pRef[i] - pTest[i]) * (pRef[i] - pTest[i]);
    }

	/* Checking for a NAN value in EnergyError */
	test =   (int *)(&EnergyError);
    temp =  *test;

    if (temp == 0x7FC00000)
    {
  		return(0);
    }


  SNR = 10 * (float32_t)log10 ((double)EnergySignal / (double)EnergyError);

  return (SNR);

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
