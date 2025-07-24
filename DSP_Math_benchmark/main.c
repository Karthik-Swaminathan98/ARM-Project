#include "main.h"

void benchmark_sqrt(void);
void benchmark_atan2(void);
void benchmark_sin_cos(void);

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
    printf("-----Starting CMSIS Math Benchmark-----\n\r");
    printf("\n\r");
    printf("CPU Clock Frequency: %lu Hz\n\r", clkFastfreq);
    printf("\n\r");

    // Perform benchmarks
//    printf("*****Benchmarking ARM SQRT *****\n\r");
//    benchmark_sqrt();
//    printf("\n\r");

//    printf("*****Benchmarking ARM ATAN2 *****\n\r");
//    benchmark_atan2();
//    printf("\n\r");

    printf("*****Benchmarking ARM SIN/COS *****\n\r");
    benchmark_sin_cos();
    printf("\n\r");

    printf("All tests are completed.\n\r");
    printf("Finish CMSIS Math Benchmark\n\r");

    return 0;
}
