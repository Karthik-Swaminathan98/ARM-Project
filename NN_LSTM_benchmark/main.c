#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <stdlib.h>
//#include "funcs_def.h"

void lstm_1();
void lstm_2();
void lstm_one_time_step();
void lstm_1_s16();
void lstm_2_s16();
void lstm_one_time_step_s16();
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
    printf("-----Starting CMSIS-LSTM Functions benchmark-----\n\r");
    printf("\n\r");
    printf("*****ARM LSTM S8*****\n\r");
    lstm_1();
    lstm_2();
    lstm_one_time_step();
    printf("\n\r");
    printf("*****ARM LSTM S16*****\n\r");
    lstm_1_s16();
    lstm_2_s16();
    lstm_one_time_step_s16();
    printf("All tests are passed.\n\r");
	printf("Finish LSTM Functions benchmark\n\r");
	return 0;

}
