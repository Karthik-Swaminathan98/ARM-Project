#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <stdlib.h>
//#include "funcs_def.h"

void fully_connected_arm_fully_connected_s8();
void fc_per_ch_arm_fully_connected_s8();
void fully_connected_mve_0_arm_fully_connected_s8();
void fully_connected_int16_arm_fully_connected_s16();
void fully_connected_int16_big_arm_fully_connected_s16();
void fc_int16_slow_arm_fully_connected_s16();


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
    printf("-----Starting CMSIS-Fully Connected Functions benchmark-----\n\r");
    printf("*****ARM Fully Connected S8*****\n\r");
    fully_connected_arm_fully_connected_s8();
    fc_per_ch_arm_fully_connected_s8();
    fully_connected_mve_0_arm_fully_connected_s8();
    printf("\n\r");
    printf("*****ARM Fully ConnectedR S16*****\n\r");
    fully_connected_int16_arm_fully_connected_s16();
    fully_connected_int16_big_arm_fully_connected_s16();
    fc_int16_slow_arm_fully_connected_s16();
	printf("Finish Fully Connected Functions benchmark\n\r");
	return 0;

}
