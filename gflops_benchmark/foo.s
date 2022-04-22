.globl cpufp_kernel_x86_avx_fp32
.globl cpufp_kernel_x86_fma_fp32

cpufp_kernel_x86_avx_fp32:
    mov $0x40000000, %rax
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm1, %ymm1, %ymm1
    vxorps %ymm2, %ymm2, %ymm2
    vxorps %ymm3, %ymm3, %ymm3
    vxorps %ymm4, %ymm4, %ymm4
    vxorps %ymm5, %ymm5, %ymm5
    vxorps %ymm6, %ymm6, %ymm6
    vxorps %ymm7, %ymm7, %ymm7
    vxorps %ymm8, %ymm8, %ymm8
    vxorps %ymm9, %ymm9, %ymm9
    vxorps %ymm10, %ymm10, %ymm10
    vxorps %ymm11, %ymm11, %ymm11
    vxorps %ymm12, %ymm12, %ymm12
.cpufp.x86.avx.fp32.L1:
    vmulps %ymm12, %ymm12, %ymm0
    vaddps %ymm12, %ymm12, %ymm1
    vmulps %ymm12, %ymm12, %ymm2
    vaddps %ymm12, %ymm12, %ymm3
    vmulps %ymm12, %ymm12, %ymm4
    vaddps %ymm12, %ymm12, %ymm5
    vmulps %ymm12, %ymm12, %ymm6
    vaddps %ymm12, %ymm12, %ymm7
    vmulps %ymm12, %ymm12, %ymm8
    vaddps %ymm12, %ymm12, %ymm9
    vmulps %ymm12, %ymm12, %ymm10
    vaddps %ymm12, %ymm12, %ymm11
    sub $0x1, %rax
    jne .cpufp.x86.avx.fp32.L1
    ret

cpufp_kernel_x86_fma_fp32:
    mov $0x40000000, %rax
    vxorps %ymm0, %ymm0, %ymm0
    vxorps %ymm1, %ymm1, %ymm1
    vxorps %ymm2, %ymm2, %ymm2
    vxorps %ymm3, %ymm3, %ymm3
    vxorps %ymm4, %ymm4, %ymm4
    vxorps %ymm5, %ymm5, %ymm5
    vxorps %ymm6, %ymm6, %ymm6
    vxorps %ymm7, %ymm7, %ymm7
    vxorps %ymm8, %ymm8, %ymm8
    vxorps %ymm9, %ymm9, %ymm9
.cpufp.x86.fma.fp32.L1:
    vfmadd231ps %ymm0, %ymm0, %ymm0
    vfmadd231ps %ymm1, %ymm1, %ymm1
    vfmadd231ps %ymm2, %ymm2, %ymm2
    vfmadd231ps %ymm3, %ymm3, %ymm3
    vfmadd231ps %ymm4, %ymm4, %ymm4
    vfmadd231ps %ymm5, %ymm5, %ymm5
    vfmadd231ps %ymm6, %ymm6, %ymm6
    vfmadd231ps %ymm7, %ymm7, %ymm7
    vfmadd231ps %ymm8, %ymm8, %ymm8
    vfmadd231ps %ymm9, %ymm9, %ymm9
    sub $0x1, %rax
    jne .cpufp.x86.fma.fp32.L1
    ret