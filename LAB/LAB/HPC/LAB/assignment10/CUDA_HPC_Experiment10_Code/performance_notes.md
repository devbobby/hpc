# Performance Measurement Notes

- Use CUDA events for kernel timing and synchronize before reading elapsed time.
- Distinguish kernel-only time from end-to-end time, which can include memory allocation and host/device transfers.
- Run multiple trials; GPU clocks and system load affect measurements.

Formulas:
- Speedup = reference execution time / optimized execution time
- Efficiency = speedup / P × 100% (interpret P according to the chosen parallel model)
- Square matrix multiplication FLOPs ≈ 2 × N^3
- GFLOPS = FLOPs / (seconds × 10^9)

Record actual measurements from the machine used. Do not present example or assumed timings as measured results.
