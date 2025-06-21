#include "cyhal.h"
#include "cybsp.h"
#include "cy_retarget_io.h"
#include "arm_math.h"
#include "core_cm4.h"
#include <math.h>
#include <stdio.h>

#define SINE_FREQ 50
#define SAMPLING_FREQ 256
#define FFT_SIZE 128

float32_t input[FFT_SIZE * 2]; // Interleaved real + imag
float32_t output[FFT_SIZE];

extern uint32_t __StackLimit;
extern uint32_t __StackTop;

void generate_sine_wave(float32_t* input, int N) {
    for (int i = 0; i < N; i++) {
        input[2 * i] = sin(2 * M_PI * SINE_FREQ * i / SAMPLING_FREQ);
        input[2 * i + 1] = 0.0f;
    }
}

void init_dwt_counters(void) {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk; // Enable trace
    DWT->CTRL |= (1 << 0);   // Enable CYCCNT
    DWT->CTRL |= (1 << 16);  // CPICNT
    DWT->CTRL |= (1 << 17);  // EXCCNT
    DWT->CTRL |= (1 << 18);  // SLEEPCNT
    DWT->CTRL |= (1 << 19);  // LSUCNT
    DWT->CTRL |= (1 << 20);  // FOLDCNT
}

void reset_dwt_counters(void) {
    DWT->CYCCNT = 0;
    DWT->CPICNT = 0;
    DWT->EXCCNT = 0;
    DWT->SLEEPCNT = 0;
    DWT->LSUCNT = 0;
    DWT->FOLDCNT = 0;
}

uint32_t estimate_instruction_count(uint32_t cycle_count) {
	printf(" DWT->CYCCNT:%lu\n\r",(unsigned long)DWT->CYCCNT);
	printf(" DWT->CPICNT:%lu\n\r",(unsigned long)DWT->CPICNT);
	printf(" DWT->EXCCNT:%lu\n\r",(unsigned long)DWT->EXCCNT);
	printf(" DWT->SLEEPCNT:%lu\n\r",(unsigned long)DWT->SLEEPCNT);
	printf(" DWT->LSUCNT:%lu\n\r",(unsigned long)DWT->LSUCNT);
	printf(" DWT->FOLDCNT:%lu\n\r",(unsigned long)DWT->FOLDCNT);

    return cycle_count
         - DWT->CPICNT
         - DWT->EXCCNT
         - DWT->SLEEPCNT
         - DWT->LSUCNT
         + DWT->FOLDCNT;
}

void fill_stack_pattern_to_sp(void) {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) {
        *p++ = 0xAAAAAAAA;
    }
}

void print_stack_info(void) {
    uint32_t stack_top = (uint32_t)&__StackTop;
    uint32_t stack_limit = (uint32_t)&__StackLimit;
    uint32_t stack_size = stack_top - stack_limit;

    printf("Stack Top    : 0x%08lX\n\r", (unsigned long)stack_top);
    printf("Stack Bottom : 0x%08lX\n\r", (unsigned long)stack_limit);
    printf("Stack Size   : %lu bytes\n\r", (unsigned long)stack_size);
}

int main(void) {
    cy_rslt_t result;

    result = cybsp_init();
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    __enable_irq();

    result = cy_retarget_io_init_fc(CYBSP_DEBUG_UART_TX, CYBSP_DEBUG_UART_RX,
                                    CYBSP_DEBUG_UART_CTS, CYBSP_DEBUG_UART_RTS,
                                    CY_RETARGET_IO_BAUDRATE);
    if (result != CY_RSLT_SUCCESS) {
        CY_ASSERT(0);
    }

    printf("Starting FFT stack and instruction usage test...\n\r");

    int N = FFT_SIZE;
    generate_sine_wave(input, N);

    arm_cfft_instance_f32 fft_instance;
    arm_cfft_init_f32(&fft_instance, N);

    // Prepare DWT counters
    init_dwt_counters();
    reset_dwt_counters();

    // Fill stack
    fill_stack_pattern_to_sp();
    register uint32_t *sp_before;
    __asm volatile ("mov %0, sp" : "=r" (sp_before));

    // Start timing
    uint32_t start_cycles = DWT->CYCCNT;

    // Execute FFT
    arm_cfft_f32(&fft_instance, input, 0, 1);

    // End timing
    uint32_t end_cycles = DWT->CYCCNT;
    uint32_t cycle_count = end_cycles - start_cycles;
    uint32_t instr_count = estimate_instruction_count(cycle_count);

    // Estimate stack usage
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp_before) {
        if (*p != 0xAAAAAAAA) break;
        p++;
    }
    uint32_t stack_used = ((uint32_t)sp_before - (uint32_t)p);

    // Compute magnitude spectrum
    arm_cmplx_mag_f32(input, output, N);

    //print_stack_info();
    printf("Stack used by arm_cfft_f32: %lu bytes\n\r", (unsigned long)stack_used);
    printf("Cycle Count: %lu cycles\n\r", (unsigned long)cycle_count);
    printf("Estimated Instruction Count: %lu\n\r", (unsigned long)instr_count);


    return 0;
}
