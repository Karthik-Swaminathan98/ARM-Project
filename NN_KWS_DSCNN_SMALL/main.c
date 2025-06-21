#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <stdlib.h>
//#include "funcs_def.h"

void layer1_arm_depthwise_conv_s8(uint32_t *total_cycles, uint32_t *total_stack);
void layer2_arm_depthwise_conv_s8(uint32_t *total_cycles, uint32_t *total_stack);
void layer3_arm_conv_s8(uint32_t *total_cycles, uint32_t *total_stack);
void layer12_arm_avgpool_s8(uint32_t *total_cycles, uint32_t *total_stack);
void layer13_arm_fully_connected_s8(uint32_t *total_cycles, uint32_t *total_stack);



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

    uint32_t total_cycles = 0;
    uint32_t total_stack = 0;

	printf("\n\r");
    printf("-----Starting KWS-DSCNN_SMALL benchmark-----\n\r");
    printf("\n\r");
    printf("*****Layers 1: DEPTHWISE_CONV_2D*****\n\r");
    layer1_arm_depthwise_conv_s8(&total_cycles, &total_stack);
    printf("\n\r");
    printf("*****Layers 2, 4, 6, 8, 10: DEPTHWISE_CONV_2D*****\n\r");
    layer2_arm_depthwise_conv_s8(&total_cycles, &total_stack);
    printf("\n\r");
    printf("*****Layers 3, 5, 7, 9, 11: CONV_2D*****\n\r");
    layer3_arm_conv_s8(&total_cycles, &total_stack);
    printf("\n\r");
    printf("*****Layers 12: AVERAGE_POOL_2D*****\n\r");
    layer12_arm_avgpool_s8(&total_cycles, &total_stack);
    printf("\n\r");
    printf("*****Layers 13: FULLY_CONNECTED*****\n\r");
    layer13_arm_fully_connected_s8(&total_cycles, &total_stack);
    // Print total results
    printf("\n\r");
    printf("-----KWS-DSCNN_SMALL benchmark complete-----\n\r");
    printf("Total Cycle Count: %lu\n\r", (unsigned long)total_cycles);
    printf("Total Stack Usage: %lu bytes\n\r", (unsigned long)total_stack);
	return 0;

}
