#include "main.h"

const int FIR_SIZES[] = {32, 64, 128, 256, 512, 1024};

RAM_FUNC void fill_stack_pattern_to_sp(void) {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp) {
        *p++ = 0xAAAAAAAA;
    }
}

RAM_FUNC void enable_cycle_counter() {
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;      // Enable trace
    DWT->CTRL |= (1 << 0);                               // CYCCNTENA
    DWT->CTRL |= (1 << 16);                              // CPICNT
    DWT->CTRL |= (1 << 17);                              // EXCCNT
    DWT->CTRL |= (1 << 18);                              // SLEEPCNT
    DWT->CTRL |= (1 << 19);                              // LSUCNT
    DWT->CTRL |= (1 << 20);                              // FOLDCNT

    DWT->CYCCNT = 0;
    DWT->CPICNT = 0;
    DWT->EXCCNT = 0;
    DWT->SLEEPCNT = 0;
    DWT->LSUCNT = 0;
    DWT->FOLDCNT = 0;
}

RAM_FUNC uint32_t read_cycle_counter(void) {
    return DWT->CYCCNT;
}

RAM_FUNC uint32_t measure_stack_usage(void) {
    register uint32_t *sp;
    __asm volatile ("mov %0, sp" : "=r" (sp));
    uint32_t *p = (uint32_t*)&__StackLimit;
    while (p < sp && *p == 0xAAAAAAAA) {
        p++;
    }
    return ((uint32_t)sp - (uint32_t)p);
}

const float32_t firCoeffs32[NUM_TAPS] = {
    -0.0018225230f, -0.0015879294f, +0.0000000000f, +0.0036977508f,
    +0.0080754303f, +0.0085302217f, -0.0000000000f, -0.0173976984f,
    -0.0341458607f, -0.0333591565f, +0.0000000000f, +0.0676308395f,
    +0.1522061835f, +0.2229246956f, +0.2504960933f, +0.2229246956f,
    +0.1522061835f, +0.0676308395f, +0.0000000000f, -0.0333591565f,
    -0.0341458607f, -0.0173976984f, -0.0000000000f, +0.0085302217f,
    +0.0080754303f, +0.0036977508f, +0.0000000000f, -0.0015879294f,
    -0.0018225230f
};


const q15_t firCoeffsQ15[NUM_TAPS_q15] = {
		2411, 4172, 5626, 6446, 6446, 5626, 4172, 2411 // Coefficients padded to make NUM_TAPS even
};
