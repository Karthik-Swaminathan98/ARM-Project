#include "main.h"

void transpose_default_arm_transpose_s8();
void transpose_3dim2_arm_transpose_s8();
void transpose_matrix_arm_transpose_s8();
void transpose_conv_1_arm_transpose_conv_s8();
void transpose_conv_2_arm_transpose_conv_s8();
void transpose_conv_3_arm_transpose_conv_s8();

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

	printf("\n\r");
    printf("-----Starting CMSIS-Transpose Functions benchmark-----\n\r");
    printf("\n\r");
    printf("CPU Clock Frequency: %lu Hz\n\r", clkFastfreq);
    printf("\n\r");

    printf("*****ARM Transpose S8*****\n\r");
    transpose_default_arm_transpose_s8();
    transpose_3dim2_arm_transpose_s8();
    transpose_matrix_arm_transpose_s8();
    printf("\n\r");

    printf("*****ARM Transpose Convolution S8*****\n\r");
    transpose_conv_1_arm_transpose_conv_s8();
    transpose_conv_2_arm_transpose_conv_s8();
    transpose_conv_3_arm_transpose_conv_s8();

    printf("All tests are passed.\n\r");
	printf("Finish Transpose Functions benchmark\n\r");
	return 0;

}
