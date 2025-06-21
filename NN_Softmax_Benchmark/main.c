#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <stdlib.h>
//#include "funcs_def.h"

void softmax_arm_softmax_s8();
void softmax_invalid_diff_min_arm_softmax_s8();
void softmax_s16_arm_softmax_s16();
void softmax_s8_s16_arm_softmax_s8_s16();


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
    printf("-----Starting CMSIS-Softmax Functions benchmark-----\n\r");
    printf("*****ARM Softmax Functions S8*****\n\r");
    softmax_arm_softmax_s8();
    softmax_invalid_diff_min_arm_softmax_s8();
    printf("\n\r");
    printf("*****ARM Softmax Functions S16*****\n\r");
    softmax_s16_arm_softmax_s16();
    printf("\n\r");
    printf("*****ARM Softmax Functions S8_S16*****\n\r");
    softmax_s8_s16_arm_softmax_s8_s16();
    printf("All tests are passed.\n\r");
	printf("Finish Softmax Functions benchmark\n\r");
	return 0;

}
