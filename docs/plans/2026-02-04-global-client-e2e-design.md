# Global Client E2E Test Design

## Overview

E2E test for the global client feature (commit 2a2e898a4) that validates the full data path through periodic CDC switchovers between two real Milvus clusters, with a mock topology REST server orchestrating the switchovers.

The test measures insert failure rate and latency impact during switchovers, with results reported both as a terminal summary and as Prometheus metrics for Grafana visualization.

## Architecture

Three components:

1. **Two real Milvus standalones** (started externally via `milvus_control -m`):
   - Cluster A (`by-dev1`) on `localhost:19530` — initial primary
   - Cluster B (`by-dev2`) on `localhost:19531` — initial secondary
   - CDC replication configured between them via `update_replicate_configuration()`

2. **Mock topology REST server** (Python `http.server`, started as pytest fixture):
   - Serves `GET /global-cluster/topology` — returns current topology with primary endpoint
   - Holds mutable state tracking which cluster is currently primary
   - Exposes `switchover()` method that:
     1. Calls `update_replicate_configuration()` on old primary first (demote)
     2. Calls `update_replicate_configuration()` on new primary second (promote)
     3. Flips the topology response (swap primary endpoint, bump version)

3. **Test client** (MilvusClient with `is_global_endpoint` patched):
   - Connects to mock topology server URL
   - `is_global_endpoint` patched to return `True` for the mock URL
   - All CRUD operations route through the global client path to the real Milvus primary

Data flow:
```
Test -> MilvusClient(global) -> Mock Topology Server -> returns primary endpoint
                              -> MilvusClient routes gRPC to real Milvus primary
```

## File Layout

```
tests/e2e_global_client/
├── conftest.py                 # pytest fixtures (mock server, metrics, clients)
├── topology_server.py          # Mock topology REST server + switchover orchestrator
├── metrics_collector.py        # Insert result recording + Prometheus exporter + report
└── test_global_client_e2e.py   # Test cases
```

### Component Responsibilities

| Component | Role |
|-----------|------|
| `topology_server.py` | HTTP server serving `/global-cluster/topology`. Holds mutable state (current primary). Exposes `switchover()` method that calls `update_replicate_configuration()` on both clusters then flips topology response. Holds two direct (non-global) MilvusClient connections to both clusters for issuing switchover API calls. |
| `metrics_collector.py` | Collects `InsertResult` and `SwitchoverEvent` into thread-safe lists. Exposes Prometheus counters/histograms via `prometheus_client`. Generates the terminal summary report at the end. |
| `conftest.py` | Fixtures: start/stop topology server, start/stop Prometheus metrics HTTP endpoint, create patched MilvusClient, configure switchover interval and test duration. |
| `test_global_client_e2e.py` | The actual test: runs insert loop + switchover ticker concurrently, asserts on failure rate and latency thresholds. |

## Test Execution Model

Two concurrent loops running in parallel:

### Insert Worker (foreground)

Continuously inserts vectors in a tight loop. Each insert is timed with `time.perf_counter()`. Records every operation:

```python
@dataclass
class InsertResult:
    timestamp: float      # wall clock time
    latency_ms: float     # operation duration
    success: bool
    error: Optional[str]  # error message if failed
```

Does NOT raise on failure — records and continues.

### Switchover Ticker (background thread)

Fires on a schedule (see Timing section). Each switchover:

1. Records switchover start time
2. Builds new `ReplicateConfiguration` with swapped topology edges
3. Calls `update_replicate_configuration()` on OLD primary first (demote)
4. Calls `update_replicate_configuration()` on NEW primary second (promote)
5. Flips internal topology state (swap primary endpoint + bump version)
6. Records switchover end time

Collects:
```python
@dataclass
class SwitchoverEvent:
    timestamp: float
    old_primary: str
    new_primary: str
    duration_ms: float
    success: bool
```

## Switchover Detail

### CDC Replication Configuration

Both clusters configured with star topology. The same config dict is sent to both clusters — only the topology edge direction changes.

- Cluster IDs: `by-dev1`, `by-dev2`
- PChannels: `{cluster_id}-rootcoord-dml_{0..15}` (16 channels)
- Token: `root:Milvus`

### Call Order

The old primary is updated first, then the new primary:

- When switching A->B (B becomes primary): call A first (demote), then B (promote)
- When switching B->A (A becomes primary): call B first (demote), then A (promote)

Reference: `/home/sheep/workspace/snippets/tests/test_cdc/testcases/update_config.py`

## Timing & Test Phases (12 minutes total)

The global client's `TopologyRefresher` interval is 5 minutes. The test uses a mixed fast+slow strategy to cover both error-triggered and background-refresh discovery paths.

### Phases

```
Phase 1: Setup (t=0)
  - Init CDC replication A->B
  - Start insert loop
  - Start Prometheus metrics endpoint

Phase 2: Fast switchovers (t=0 -> t=4min)
  - Switchover every 2 minutes
  - 2 switchovers: A->B, B->A
  - Client discovers changes via error-triggered refresh (UNAVAILABLE)
  - Tests: error recovery path, reconnect speed

Phase 3: Slow switchovers (t=4min -> t=12min)
  - Switchover every 4 minutes
  - 2 switchovers: A->B, B->A
  - Client has time for background TopologyRefresher to fire (5-min interval)
  - Tests: background refresh discovery, steady-state after recovery

Phase 4: Teardown + Report (t=12min)
  - Stop insert loop
  - Print terminal report
  - Assert thresholds
  - Prometheus endpoint stays up briefly for final scrape
```

### Timeline

```
t=0      t=2min   t=4min         t=8min              t=12min
|--------|--------|--------------|---------------------|
 A->B     B->A     A->B           B->A                 end
 fast     fast     slow           slow
```

4 switchovers total. Ends on B->A so original primary (A) is restored.

## Monitoring

### Prometheus & Grafana

The existing monitoring stack at `/home/sheep/workspace/snippets/monitor` already covers both clusters:

- Cluster A (`by-dev1`): metrics on port `19091`, CDC on `29091` (scraped by `standalone` job)
- Cluster B (`by-dev2`): metrics on port `19092`, CDC on `29092` (scraped by `cluster` job)

Start the monitoring stack before running the test.

### Test Process Metrics

The test process exposes Prometheus metrics on port `9200`:

- `global_test_insert_latency_seconds` (Histogram)
- `global_test_insert_total` (Counter, labels: `status=success|fail`)
- `global_test_switchover_total` (Counter)
- `global_test_switchover_duration_seconds` (Histogram)
- `global_test_current_primary` (Gauge with label)

Add the scrape target to `prometheus.yml`:
```yaml
- job_name: 'global_e2e_test'
  static_configs:
    - targets: ['host.docker.internal:9200']
```

### Terminal Report

Printed at test end:

```
================== Global Client E2E Test Report ==================
Duration: 720.2s | Switchovers: 4 | Strategy: 2 fast(2min) + 2 slow(4min)
Clusters: by-dev1 (localhost:19530) <-> by-dev2 (localhost:19531)

--- Insert Summary ---
Total: 21432 | Success: 21427 | Failed: 5 | Failure Rate: 0.02%

--- Latency (ms) ---
                  Steady State    Switchover Window (+/-5s)
  p50:            12.3            45.6
  p95:            28.1            312.4
  p99:            55.2            1205.8
  max:            89.4            2841.1

--- Switchover Events ---
  #1 [t=0.0s]    by-dev1 -> by-dev2  duration=1.23s  window_failures=1
  #2 [t=120.0s]  by-dev2 -> by-dev1  duration=1.15s  window_failures=0
  #3 [t=240.2s]  by-dev1 -> by-dev2  duration=1.31s  window_failures=2
  #4 [t=480.1s]  by-dev2 -> by-dev1  duration=1.08s  window_failures=2

--- Prometheus Metrics ---
  Endpoint: http://localhost:9200/metrics
================================================================
```

## Test Assertions

### Hard assertions (test fails if violated)

- Insert failure rate < 5% overall
- No insert failures outside of switchover windows (+/-5s around each switchover event)
- All switchover API calls succeed
- Global client reconnects to new primary within 30s after each switchover

### Soft assertions (logged as warnings, don't fail)

- p99 latency during switchover windows < 5s
- p50 latency during steady state < 100ms

### Configurable Thresholds (pytest CLI)

- `--switchover-interval`: seconds between fast switchovers (default: 120)
- `--test-duration`: total test run time in seconds (default: 720)
- `--max-failure-rate`: maximum allowed failure rate (default: 0.05)
- `--switchover-window`: seconds around switchover to tolerate failures (default: 5)
- `--primary-port`: initial primary port (default: 19530)
- `--standby-port`: initial standby port (default: 19531)
- `--metrics-port`: Prometheus metrics endpoint port (default: 9200)

## Prerequisites

Before running the test:

1. Docker network `milvus_inf` exists
2. Infrastructure running (etcd, minio) via `docker-compose-rmq.yml`
3. Monitoring stack running (Prometheus, Grafana) via `monitor/docker-compose.yaml`
4. Two Milvus clusters running via `milvus_control -m start_milvus`
5. Python environment with `pymilvus` and `prometheus_client` installed
