# Burn-in

Simple PyTorch matmul runner to test all CPU cores and CUDA-compatible GPUs on a system.

Runs one set of replicates per device. A device means one CPU core, or one GPU. All CPU and GPU sets are run in parallel, replicates are run sequentially for a given device. Produces a `results.csv` file by default.

## Usage

Cores and GPUs are detected automatically. Warmups are performed automatically to ensure better consistency.

```bash
python burn.py
```

Supply a custom config file with `-c` or `--config-file`.

```bash
python burn.py --config-file /path/to/config.yml
```

Supply a desired path for output csv file with `-o` or `--output-file`.

```bash
python burn.py --output-file /path/to/reults.csv
```

## Config

Sane values for consumer desktop computers in 2025 are provided, runtime should be around 20 minutes. If you are using more powerful GPUs, consider increasing the matrix size.

- `matrix_size` determines the size of the square matrices being multiplied. Remember, matrix multiplication is `O(n^3)` runtime, so be careful about increasing this value.
- `replicates` determines how many repeats to run of the same multiplication. Runtime contribution is `O(n)` for this variable.

Total runtime will be `O(matrix_size^3 * replicates)`.
