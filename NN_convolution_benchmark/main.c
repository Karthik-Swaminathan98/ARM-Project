#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include <stdlib.h>
//#include "funcs_def.h"

void basic_arm_convolve_s8();
void conv_3x3_dilation_5x5_input_arm_convolve_s8();
void conv_2x2_dilation_arm_convolve_s8();
void basic_arm_convolve_s16();
void int16xint8_dilation_1_arm_convolve_s16();
void int16xint8xint32_1_arm_convolve_s16();
void depthwise_2_arm_depthwise_conv_s8();
void depthwise_mult_batches_arm_depthwise_conv_s8();
void depthwise_dilation_arm_depthwise_conv_s8();
void dw_int16xint8_arm_depthwise_conv_s16();
void dw_int16xint8_mult4_arm_depthwise_conv_s16();
void dw_int16xint8_dilation_arm_depthwise_conv_s16();

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
    printf("-----Starting CMSIS-ConvolutionFunctions benchmark-----\n\r");
    printf("\n\r");
    printf("*****ARM CONVOLVE WRAPPER S8*****\n\r");
    basic_arm_convolve_s8();
    conv_2x2_dilation_arm_convolve_s8();
    conv_3x3_dilation_5x5_input_arm_convolve_s8();
    printf("\n\r");
    printf("*****ARM CONVOLVE WRAPPER S16*****\n\r");
    basic_arm_convolve_s16();
    int16xint8_dilation_1_arm_convolve_s16();
    int16xint8xint32_1_arm_convolve_s16();
    printf("\n\r");
    printf("*****ARM DEPTHWISE CONVOLVE WRAPPER S8*****\n\r");
    depthwise_2_arm_depthwise_conv_s8();
    depthwise_mult_batches_arm_depthwise_conv_s8();
    depthwise_dilation_arm_depthwise_conv_s8();
    printf("\n\r");
    printf("*****ARM DEPTHWISE CONVOLVE WRAPPER S16*****\n\r");
    dw_int16xint8_arm_depthwise_conv_s16();
    dw_int16xint8_mult4_arm_depthwise_conv_s16();
    dw_int16xint8_dilation_arm_depthwise_conv_s16();
	printf("Finish ConvolutionFunctions benchmark\n\r");
	return 0;

}
