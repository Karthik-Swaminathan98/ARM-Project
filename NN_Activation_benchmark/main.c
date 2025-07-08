#include "main.h"

void relu6_arm_relu6_s8(void);
void activ_arm_nn_activation_s16(void);

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
    printf("-----Starting CMSIS-Activation Functions Benchmark-----\n\r");
    printf("\n\r");
    printf("CPU Clock Frequency: %lu Hz\n\r", clkFastfreq);
    printf("\n\r");

    // Perform benchmarks
    printf("*****ARM RELU S8*****\n\r");
    relu6_arm_relu6_s8();
    printf("\n\r");

    printf("*****ARM NN ACTIVATION S16*****\n\r");
    activ_arm_nn_activation_s16();

    printf("All tests are passed.\n\r");
    printf("Finish Activation Functions Benchmark\n\r");

    return 0;
}
