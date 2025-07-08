#include "main.h"

void enable_cycle_counter() {
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

uint32_t read_cycle_counter(void) {
    return DWT->CYCCNT;
}

void fill_stack_pattern_to_sp(void) {
    register uint32_t *sp;
    __asm volatile("mov %0, sp" : "=r"(sp));

    uint32_t *p = (uint32_t *)&__StackLimit;
    while (p < sp) {
        *p++ = 0xAAAAAAAA;
    }
}

uint32_t measure_stack_usage(void) {
    register uint32_t *sp;
    __asm volatile("mov %0, sp" : "=r"(sp));

    uint32_t *p = (uint32_t *)&__StackLimit;
    while (p < sp) {
        if (*p != 0xAAAAAAAA) {
            break;
        }
        p++;
    }

    return ((uint32_t)sp - (uint32_t)p); // Stack usage in bytes
}

void generate_rand_s8(int8_t *src, int length) {
    enable_cycle_counter();
    srand(read_cycle_counter());
    for (int i = 0; i < length; i++) {
        src[i] = (int8_t)((rand() % 256) - 128); // Range: [-128, 127]
    }
}

void generate_rand_s16(int16_t *src, int length) {
    enable_cycle_counter();
    srand(read_cycle_counter());
    for (int i = 0; i < length; i++) {
        src[i] = (int16_t)((rand() % 65536) - 32768); // Range: [-32768, 32767]
    }
}
