# Rover: A Resource-Efficient Framework for On-Device Vector Search via Mobile SoC Acceleration

Rover is a graph-based ANNS framework for on-device vector search via mobile SoC acceleration. It supports ANNS using aggressive scalar quantization (SQ) and product quantization (PQ) with a high compression ratio while preserving accuracy. A distance refinement strategy, fine-grained asynchronous pipeline, and neighborhood-aware storage layout are proposed to jointly optimize memory, computation, and I/O demands, achieving comparable performance to in-memory ANNS methods with limited memory. A Python entry point on the host coordinates data preparation, optional graph layout optimization, transfers, device execution, and result collection. It employs Qualcomm Snapdragon SoCs as the prototype platform. 

## Table of Contents

- [Introduction](#introduction)
- [Build Instructions](#build-instructions)
- [Usage](#usage)
- [Parameters](#parameters)
- [Outputs and Results](#outputs-and-results)

## Introduction

Rover provides two main workflows:

- **Search:** Reuse an existing index, or prepare one from an existing graph and raw vectors with optional layout optimization → search on the device → collect results.
- **Construction:** Raw vectors → prepare SQ/PQ codes and residuals → build a graph or insert vectors on the device → retrieve the complete index → search with that index.

An existing index is reused when its configuration matches the request. When graph layout optimization is enabled for a new index, Rover reorders the graph and vectors before generating the corresponding codes and residuals.

| Branch | Features |
|---|---|
| `main` | Import existing graphs, optimize graph and residual layouts, and search with SQ/PQ |
| `graph-build-insert` | All features of `main`, plus graph construction from raw vectors, graph rebuilding, streaming insertion, and checkpoint recovery |

The components run in the following locations:

| Execution location | Programs and responsibilities |
|---|---|
| Linux host or server | `Rover_walkthrough.py` accepts commands; `tools/` prepares codes, residuals, and indexes, coordinates transfers, and collects results; host layout programs optimize graph and residual layouts |
| Android CPU | `Rover` executes search or construction tasks and calls the DSP through `libcalculator.so` |
| Hexagon DSP | `libcalculator_skel.so` performs DSP computation |

## Build Instructions

All experiments for this project were conducted on **Qualcomm SM8650 (Snapdragon 8 Gen 3)** using **Qualcomm Hexagon SDK 6.4.0.1**.

### Prerequisites

- A Linux host, Python 3.10+, and a C/C++ compiler supporting C11/C++17.
- CMake 3.20+. The host layout tools can use OpenMP.
- Python dependencies: NumPy and FAISS CPU. See [requirements.txt](requirements.txt) for version requirements.
- Qualcomm Hexagon SDK 6.4.0.1. The current build script uses the SDK's CMake 3.28.3, Android NDK r25c/API 26, and Hexagon Tools 19.0.04/v75.
- For device execution, ADB and a Qualcomm Android device that supports FastRPC and can run these binaries.

### Compile on Linux

From the Rover repository root, install the Python dependencies and build:

```sh
python3 -m pip install -r requirements.txt
python3 scripts/build.py --sdk /path/to/Hexagon_SDK/6.4.0.1 --target all --jobs 4
```

- `--sdk` specifies the SDK root directory. It can also be set through the `HEXAGON_SDK_ROOT` environment variable.
- `--target` selects `host`, `android`, `hexagon`, or `all`. The default is `all`.
- `--jobs` sets the number of parallel build jobs. The default is 4.
- `config/capabilities.json` defines the current branch's capabilities. Graph construction is enabled automatically on the construction branch.

Building only the host layout tools does not require the SDK:

```sh
python3 scripts/build.py --target host --jobs 4
```

| Output directory | Build artifacts |
|---|---|
| `dist/host/` | `rover-layout`, `rover-residual-layout` |
| `dist/android/` | `Rover`, `libcalculator.so` |
| `dist/hexagon/` | `libcalculator_skel.so` |

Rebuild both Android and Hexagon after switching feature branches or changing the shared macros in `config/rover_limits.h`, the RPC interface, or the DSP architecture. Data paths, SQ/PQ representations, residual bit widths, and search budgets are configured through runtime arguments.

## Usage

Run all commands below on the host through `Rover_walkthrough.py` in the repository root. Replace `DEVICE` with the ADB device serial number and replace the example paths with actual paths on the host. You can omit `--serial` when only one ADB device is connected.

The script automatically transfers the programs and data required for each task. Path arguments refer to host paths by default. `--device-dir` specifies the device working directory separately and defaults to `/data/local/tmp/Rover`. Relative paths are resolved from the directory where the command is invoked.

### Prepare Input Data

| Input | Format and requirements |
|---|---|
| Raw vectors, `--base` | `fvecs`: each row contains a little-endian INT32 dimension followed by an FP32 vector. Supported dimensions are 1..1024 |
| Query vectors, `--query` | `fvecs`; the dimension must match the index |
| Ground-truth neighbors, `--gt` | Optional `ivecs` for recall evaluation: each row contains a little-endian INT32 neighbor count followed by INT32 IDs |
| Existing graph, `--graph` | Select `rover`, `dense`, `nsg`, or `nndescent` with `--graph-format`; node IDs must correspond to row IDs in the raw vectors |
| Existing index, `--index-dir` | A Rover index directory containing `index.json` and the data files it references |

A `dense` graph is a row-major INT32 adjacency table without a file header. `nndescent` refers to the binary format supported by Rover's reader, with a header, neighbor IDs, and distances. NSG graphs can be generated with [NSG](https://github.com/ZJULearning/nsg). When importing NSG, `--graph-degree` must accommodate the actual neighbor count of every row. By default, the entry point is read from the file header.

### Search an Existing Index

```sh
python3 Rover_walkthrough.py search --serial DEVICE \
  --index-dir /path/to/index \
  --query /path/to/query.fvecs \
  --top-k 10 --pool 128 --entry-points 32 \
  --results-dir /path/to/search-run
```

- `--index-dir` specifies the complete index. The encoding type, bit widths, and PQ grouping are read from the index.
- `--top-k` sets the number of neighbors returned per query.
- `--pool` sets the search candidate pool size and must be at least `top-k`.
- `--entry-points` sets the number of search entry points.
- `--results-dir` specifies a new results directory. If omitted, results are written to `runs/<job-id>/`.

Add `--gt /path/to/groundtruth.ivecs` to evaluate recall. To enable final reranking with reconstructed vectors, add `--refine on --refine-factor 2`, or use `--refine on --refine-all` to rerank all available candidates.

### Import an Existing Graph and Search

The following example imports an NSG graph, generates SQ INT8 base codes, INT4 first-level residuals, and an INT4 third level, then searches with graph layout optimization enabled:

```sh
python3 Rover_walkthrough.py search --serial DEVICE \
  --base /path/to/base.fvecs \
  --graph /path/to/input.nsg --graph-format nsg --graph-degree 16 \
  --output-index-dir /path/to/new-sq-index \
  --query /path/to/query.fvecs \
  --method sq --sq-bits 8 --residual-bits 4 --third-residual on \
  --reorder on --num-clusters 256 --cn on --mnn on \
  --correction online --refine on --refine-all
```

Use a new, empty directory for `--output-index-dir`. Adjust `graph-degree` to match the input graph. Use `--reorder off` to disable graph layout optimization and `--third-residual off` to disable the third level.

To change an existing index's encoding or layout, provide the raw vectors, a graph with matching row IDs, `--reencode`, and a new `--output-index-dir`. The resulting index can then be reused through `--index-dir`.

### Build an Index from Raw Vectors

Switch to the construction branch and rebuild:

```sh
git checkout graph-build-insert
python3 scripts/build.py --sdk /path/to/Hexagon_SDK/6.4.0.1 --target all --jobs 4
```

The following example builds an index from an empty graph using streaming insertion, PQ4 base codes, INT8 first-level residuals, and an INT4 third level:

```sh
python3 Rover_walkthrough.py build-index --serial DEVICE \
  --base /path/to/base.fvecs --output-index-dir /path/to/new-pq-index \
  --method pq --pq-bits 4 --pq-compression 64 \
  --residual-bits 8 --third-residual on \
  --build-method stream-insert --graph-degree 16 \
  --build-pool 128 --build-candidates 16 --entry-points 32 \
  --results-dir /path/to/build-run
```

- `--build-method` selects `stream-insert`, `search-build`, or `rnn-descent`.
- `--graph-degree` sets the output graph width.
- `--build-pool` and `--build-candidates` set construction candidate budgets. For `stream-insert` and `search-build`, they must satisfy `graph-degree ≤ build-candidates ≤ build-pool`.
- `--pq-compression 64` specifies a target compression ratio of 64 relative to FP32 for the base codes. For example, 128 dimensions with PQ4 at 64× compression use 16 subspaces and 8 bytes of base codes per vector, excluding residuals, the graph, and codebooks.

After construction, search with `search --index-dir /path/to/new-pq-index`. To rebuild a graph, run `build-index` again with the raw vectors, a new output directory, and the desired construction parameters. Apply graph layout optimization and B/C/D residual layouts after construction through the existing-graph import workflow.

### Prepare Codes Separately from Construction

`prepare` prepares index data on the host only. Generate codes and residuals first, then reuse them for construction:

```sh
python3 Rover_walkthrough.py prepare \
  --base /path/to/base.fvecs --output-index-dir /path/to/prepared-index \
  --method sq --sq-bits 4 --residual-bits 8 --third-residual on

python3 Rover_walkthrough.py build-index --serial DEVICE \
  --index-dir /path/to/prepared-index --base /path/to/base.fvecs \
  --build-method stream-insert --graph-degree 16
```

Prepared data without a graph requires graph construction before it can be searched. Construction must use exactly the same raw vectors as preparation. You can also pass an existing graph and its parameters to `prepare` to produce a complete search index on the host.

### Insert Vectors into an Existing Graph

On `graph-build-insert`, extend a parent graph to produce an index that includes new vectors:

```sh
python3 Rover_walkthrough.py insert --serial DEVICE \
  --parent-index /path/to/parent-index \
  --base /path/to/parent_plus_new.fvecs --output-index-dir /path/to/child-index \
  --method sq --sq-bits 4 --residual-bits 4 --third-residual on \
  --build-method stream-insert --graph-degree 16
```

`--base` must contain all parent vectors as an unchanged prefix in their original order, followed by the new vectors. The parent index must contain a Rover graph that uses original IDs and has not undergone graph reordering. Its dimension and graph width must match the new index. The parent index is preserved, and the complete new index is written to `--output-index-dir`.

### Resume Interrupted Streaming Construction

`stream-insert` supports checkpoints when building from an empty graph. To resume the PQ construction example above:

```sh
python3 Rover_walkthrough.py resume --serial DEVICE \
  --index-dir /path/to/new-pq-index --base /path/to/base.fvecs \
  --checkpoint /path/to/build-run/insert.checkpoint \
  --method pq --pq-bits 4 --pq-compression 64 \
  --residual-bits 8 --third-residual on \
  --build-method stream-insert --graph-degree 16 \
  --build-pool 128 --build-candidates 16 --entry-points 32
```

`--index-dir` points to the prepared directory whose graph construction is incomplete. `--checkpoint` points to the checkpoint retrieved to the host. Recovery requires the same data, encoding, construction parameters that affect recovery state, and a matching Rover build. Batch insertion into a parent graph does not support this recovery protocol.

## Parameters

### SQ and PQ

The base encoding, first-level residual bit width, and third-level switch are configured independently:

| Base encoding | First-level residual | Third level disabled | Third level enabled |
|---|---|---|---|
| SQ INT4 | INT4 | 4 + 4 | 4 + 4 + 4 |
| SQ INT4 | INT8 | 4 + 8 | 4 + 8 + 4 |
| SQ INT8 | INT4 | 8 + 4 | 8 + 4 + 4 |
| SQ INT8 | INT8 | 8 + 8 | 8 + 8 + 4 |
| PQ4 or PQ8 | INT4 or INT8 | PQ + R4/R8 | PQ + R4/R8 + INT4 |

| Parameter | Description |
|---|---|
| `--method` | Base encoding: `sq` or `pq`; defaults to `sq` for a new index |
| `--sq-bits` | SQ base code bit width: 4 or 8; default 4 |
| `--pq-bits` | PQ code bit width per subspace: 4 or 8; default 8 |
| `--pq-m` | Number of PQ subspaces, M; must divide the vector dimension |
| `--pq-subdim` | Dimension of each PQ subspace; must divide the vector dimension |
| `--pq-compression` | Target compression ratio for the base codes relative to FP32; must yield a valid integer M |
| `--pq-train-samples`, `--pq-train-iters` | Maximum PQ training samples and training iterations; defaults 100000 and 20 |
| `--residual-bits` | First-level residual bit width: 4 or 8; default 4 |
| `--third-residual` | `off` or `on`, controlling the INT4 third level; disabled by default for a new index and inherited from an existing index |
| `--seed`, `--host-threads` | Random seed and thread count for host preparation tools; defaults 42 and 8 |

Specify at most one of `--pq-m`, `--pq-subdim`, and `--pq-compression`. If none is specified for a new index, M defaults to dimension / 16 when the dimension is divisible by 16, or to the dimension otherwise. PQ compression ratios exclude the graph, residuals, and codebooks. PQ4 also includes half-byte padding when M is odd.

For an existing index, its metadata determines the encoding type, bit widths, and PQ grouping. The third-level switch controls preparation, transfer, and loading of third-level data. If an index does not contain that data, enabling the third level requires preparing the index again.

### Online Correction and Final Reranking

- `--correction online` enables online residual correction by default. `--correction off` searches using only the base codes.
- During search, `--refine on` enables final reranking with reconstructed vectors; the default is `off`. Reranking requires `--correction online`.
- `--refine-factor N` reranks at most N × Top-K candidates and defaults to 2. `--refine-all` reranks all available candidates instead. The two options are mutually exclusive.
- `--third-residual on` provides third-level data and is controlled separately from `--refine`. With the third level disabled, final reranking can still use the base codes and first-level residuals.

### Search Parameters

| Parameter | Description and range |
|---|---|
| `--search-mode` | `sqmd` or `db-pair`; default `sqmd`. `db-pair` supports SQ |
| `--top-k` | Neighbors returned per query; default 10, range 1..100 |
| `--pool` | Search candidate pool size; default 128, range Top-K..2000 |
| `--entry-points` | Number of search entry points; default 32, no greater than the dataset size |
| `--expand-count` | Search expansion count; default 32, range 1..32 |
| `--lanes`, `--source-window` | Execution concurrency settings; defaults 4 and 1, with supported values 1..8 and 1/2 respectively |
| `--first-stage-factor` | Candidate budget multiplier for the first stage; default 3 |
| `--storage` | First-level residual storage: `disk` uses a block cache, and `memory` keeps residuals in memory; default `disk` |
| `--cache-pages`, `--refine-cache-pages` | Residual and reranking cache page counts; default 2048 each. Values must be powers of two in 1..65536; each page is 4096 B |
| `--split-distance` | `off` or `on`; disabled by default. Requires SQ4, online correction, and a prepared correction bias |

Top-K must not exceed the dataset size. Both the first-stage and reranking candidate multipliers, when multiplied by Top-K, must stay within the current capacity of 2000. The actual candidate count is limited by the search results. Cache memory usage also grows with the number of concurrent workspaces.

### Construction Parameters

| Parameter | Description and applicable methods |
|---|---|
| `--graph-degree` | Output graph width; default 16, smaller than the dataset size |
| `--build-pool`, `--build-candidates` | Construction search pool size and candidate count; defaults 128 and 16. The pool size is at most 2000 |
| `--build-iterations` | Construction iterations; default 1, used by `search-build` and `rnn-descent` |
| `--build-threads` | Default 1. `stream-insert` requires 1; `search-build` supports 1..4; `rnn-descent` supports 1..64 |
| `--build-refine-candidates` | Candidate reranking budget during construction; 0 uses build-pool. A nonzero value must be between build-candidates and build-pool |
| `--reverse-candidates` | Reverse candidate budget; default 0, no greater than build-candidates |
| `--backlink-threads` | Threads for backlink processing during streaming insertion; default 1, range 1..4 |
| `--checkpoint-every` | Checkpoint interval for streaming construction; default 1000000 points |
| `--rnn-s`, `--rnn-t1` | RNN-Descent parameters; defaults 16 and 4. rnn-s must not exceed the graph width |
| `--build-distance` | FP32 distance computation for RNN-Descent: `cpu` or `dsp`; default `cpu` |
| `--cooldown-every`, `--cooldown-ms` | Periodic cooldown interval and duration for `search-build`; the interval defaults to 0, disabling cooldown |

RNN-Descent builds the graph with FP32 distances. SQ/PQ parameters determine the search representation generated alongside it. Periodic cooldown requires the `RETRIEVAL_ENABLE_BUILD_PAUSE` compile-time macro to be enabled.

### Graph and Residual Layout Parameters

| Parameter | Description |
|---|---|
| `--reorder` | `off` or `on`, controlling layout optimization for an existing graph and its vectors; disabled by default |
| `--num-clusters`, `--cluster-iters` | Cluster count and iterations for graph layout optimization. The cluster count defaults to the integer part of the square root of the dataset size; iterations default to 20 |
| `--cn`, `--mnn` | Graph layout optimization switches; both default to `on`. When `--reorder on`, at least one must remain `on` |
| `--residual-layout` | `a`, `b`, `c`, or `d`; default `a`. B/C/D apply to SQ residuals and require an existing graph |

When residual layouts are changed, the first-level residuals and third level use the same corresponding mapping. Detailed B/C/D settings are in `config/rover_limits.h`; rebuild the host layout tools after changing them.

### Parameter Profiles

`--profile` accepts a preset name from `config/profiles/` or a path to a configuration file. For example, import a new index using SQ8 + INT8 residuals + an INT4 third level:

```sh
python3 Rover_walkthrough.py search --serial DEVICE \
  --profile sq8-r8-third \
  --base /path/to/base.fvecs \
  --graph /path/to/input.nsg --graph-format nsg --graph-degree 16 \
  --output-index-dir /path/to/profile-index \
  --query /path/to/query.fvecs --top-k 10 --pool 256
```

Other presets include `sq4-r4` and `pq4-64x-third`. Parameter precedence is defaults → profile → explicit command-line arguments. Representation constraints from an existing index still apply.

## Outputs and Results

| Output location | Contents |
|---|---|
| `--output-index-dir` | The generated graph, codes, residuals, and `index.json`; the complete directory can be reused for search |
| `--results-dir` or `runs/<job-id>/` | Request parameters and device logs. Successful search tasks include `results.jsonl` and available statistics; construction tasks retrieve available checkpoints |
| `--device-dir` on the device | Programs and task-specific data working directories |

Neighbor IDs in `results.jsonl` refer to the original data rows. IDs are mapped back even if graph layout optimization was applied. A new output index directory must be empty, and a results directory must not already exist.

To transfer build artifacts separately:

```sh
python3 scripts/push.py --serial DEVICE
```

To list all arguments or print the parsed request only:

```sh
python3 Rover_walkthrough.py --help
python3 Rover_walkthrough.py search --index-dir /path/to/index \
  --query /path/to/query.fvecs --dry-run
```

`--dry-run` prints the request parameters only. It does not validate data integrity, prepare data, or execute device tasks.
