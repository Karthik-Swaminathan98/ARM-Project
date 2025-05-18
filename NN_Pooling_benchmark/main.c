#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <stdlib.h>
//#include "funcs_def.h"

void avgpooling_arm_avgpool_s8();
void avgpooling_int16_arm_avgpool_s16();


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
    printf("Starting CMSIS-Pooling Functions benchmark\n\r");
    avgpooling_arm_avgpool_s8();
    avgpooling_int16_arm_avgpool_s16();
    printf("All tests are passed.\n\r");
	printf("Finish Pooling Functions benchmark\n\r");
	return 0;

}
