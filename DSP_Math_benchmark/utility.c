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

