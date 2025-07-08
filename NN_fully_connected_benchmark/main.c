#include "main.h"

void fully_connected_arm_fully_connected_s8();
void fc_per_ch_arm_fully_connected_s8();
void fully_connected_mve_0_arm_fully_connected_s8();
void fully_connected_int16_arm_fully_connected_s16();
void fully_connected_int16_big_arm_fully_connected_s16();
void fc_int16_slow_arm_fully_connected_s16();

uint32_t clkFastfreq = 0;

RAM_FUNC int main(void)
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

    // Print system details
    printf("\n\r");
    printf("-----Starting CMSIS-Fully Connected Functions benchmark-----\n\r");
    printf("\n\r");
    printf("CPU Clock Frequency: %lu Hz\n\r", clkFastfreq);
    printf("\n\r");

    // Perform benchmarks
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
