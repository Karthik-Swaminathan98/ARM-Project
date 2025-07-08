#include "main.h"

const int FFT_SIZES[FFT_SIZES_COUNT] = {32, 64, 128, 256, 512, 1024};

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

RAM_FUNC void calculate_averages(uint32_t* cycle_counts, uint32_t* instr_counts, float* exec_time_us, uint32_t* stack_usages, int num_executions) {
    uint32_t total_cycles = 0, total_instr = 0, total_stack_used = 0;
    float total_exec_time_us = 0;

    for (int i = 0; i < num_executions; i++) {
        total_cycles += cycle_counts[i];
        total_instr += instr_counts[i];
        total_stack_used += stack_usages[i];
        total_exec_time_us += exec_time_us[i];
    }

    printf("\nAverages across %d executions:\n\r", num_executions);
    printf("Cycle Count = %lu\n\r", total_cycles / num_executions);
    printf("Estimated Instructions = %lu\n\r", total_instr / num_executions);
    printf("Execution Time = %.2f us\n\r", total_exec_time_us / num_executions);
    printf("Stack Used = %lu bytes\n\r", total_stack_used / num_executions);
}
