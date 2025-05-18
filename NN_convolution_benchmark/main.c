#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <stdlib.h>
//#include "funcs_def.h"

void basic_arm_convolve_s8();
void basic_arm_convolve_s16();
void depthwise_2_arm_depthwise_conv_s8();
void dw_int16xint8_arm_depthwise_conv_s16();

int main(void)
{
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
	printf("\n\r");
    printf("Starting CMSIS-ConvolutionFunctions benchmark\n\r");
    basic_arm_convolve_s8();
    basic_arm_convolve_s16();
    depthwise_2_arm_depthwise_conv_s8();
    dw_int16xint8_arm_depthwise_conv_s16();
    printf("All tests are passed.\n\r");
	printf("Finish ConvolutionFunctions benchmark\n\r");
	return 0;

}
