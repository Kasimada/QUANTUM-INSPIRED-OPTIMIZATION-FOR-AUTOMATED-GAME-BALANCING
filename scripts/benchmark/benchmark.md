# Game Balancing - Architecture Refactoring Benchmark Report

This report compares the performance of the evaluation system before and after the Strategy Pattern and Dataclass refactoring.

## CPU Benchmarks

| Workers | Baseline Runtime (s) | Final Runtime (s) | Baseline Throughput | Final Throughput |
|---|---|---|---|---|
| 1 | 0.5296 | 0.4537 | 188.84 | 220.40 |
| 2 | 3.7679 | 3.6218 | 26.54 | 27.61 |
| 4 | 4.1788 | 4.2879 | 23.93 | 23.32 |
| 8 | 6.0878 | 6.1652 | 16.43 | 16.22 |