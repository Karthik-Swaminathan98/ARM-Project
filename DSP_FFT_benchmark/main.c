#include "main.h"

void benchmark_f32(void);
void benchmark_q15(void);

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
    printf("-----Starting CMSIS FFT Benchmark-----\n\r");
    printf("\n\r");
    printf("CPU Clock Frequency: %lu Hz\n\r", clkFastfreq);
    printf("\n\r");

    // Perform benchmarks
    printf("*****Benchmarking ARM CFFT F32*****\n\r");
    benchmark_f32();
    printf("\n\r");

    printf("*****Benchmarking ARM CFFT Q15*****\n\r");
    benchmark_q15();
    printf("\n\r");

    printf("All tests are completed.\n\r");
    printf("Finish CMSIS FFT Benchmark\n\r");

    return 0;
}
