#include "main.h"

void avgpooling_arm_avgpool_s8();
void avgpooling_1_arm_avgpool_s8();
void avgpooling_2_arm_avgpool_s8();
void avgpooling_int16_arm_avgpool_s16();
void avgpooling_int16_1_arm_avgpool_s16();
void avgpooling_int16_2_arm_avgpool_s16();

uint32_t clkFastfreq = 0;

int main(void)
{
	__disable_irq();  // Disable all interrupts
    cy_rslt_t result;

    // Initialize the device and board peripherals
    result = cybsp_init();
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    // Initialize retarget-io to use the debug UART port
    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                                    CYBSP_DEBUG_UART_CTS, CYBSP_DEBUG_UART_RTS, CY_RETARGET_IO_BAUDRATE);

    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    // Get the clock frequency
    clkFastfreq = Cy_SysClk_ClkFastGetFrequency();

	printf("\n\r");
    printf("-----Starting CMSIS-Pooling Functions benchmark-----\n\r");
    printf("\n\r");
    printf("CPU Clock Frequency: %lu Hz\n\r", clkFastfreq);
    printf("\n\r");

    printf("*****ARM Average Pooling S8*****\n\r");
    avgpooling_arm_avgpool_s8();
    avgpooling_1_arm_avgpool_s8();
    avgpooling_2_arm_avgpool_s8();
    printf("\n\r");

    printf("*****ARM Average Pooling S16*****\n\r");
    avgpooling_int16_arm_avgpool_s16();
    avgpooling_int16_1_arm_avgpool_s16();
    avgpooling_int16_2_arm_avgpool_s16();

    printf("All tests are passed.\n\r");
	printf("Finish Pooling Functions benchmark\n\r");
	return 0;

}
