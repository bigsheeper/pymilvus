# Global Client E2E Test Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Build an E2E test that validates global client behavior during periodic CDC switchovers between two real Milvus clusters, measuring insert failure rate and latency.

**Architecture:** Mock topology REST server orchestrates switchovers between two Milvus standalones. A patched MilvusClient connects through the global client path. Continuous inserts run while switchovers happen on a mixed fast+slow schedule. Results are reported via terminal and Prometheus metrics.

**Tech Stack:** pytest, http.server, threading, prometheus_client, pymilvus MilvusClient

**Design Doc:** `docs/plans/2026-02-04-global-client-e2e-design.md`

---

### Task 1: Create metrics_collector.py — Data Models and Collection

**Files:**
- Create: `tests/e2e_global_client/__init__.py`
- Create: `tests/e2e_global_client/metrics_collector.py`

**Step 1: Create the package init file**

```python
# tests/e2e_global_client/__init__.py
# empty
```

**Step 2: Write metrics_collector.py with data models, thread-safe collection, Prometheus export, and report generation**

```python
# tests/e2e_global_client/metrics_collector.py
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
from prometheus_client import Counter, Gauge, Histogram, start_http_server


@dataclass
class InsertResult:
    timestamp: float  # wall clock (time.time())
    latency_ms: float  # operation duration in ms
    success: bool
    error: Optional[str] = None


@dataclass
class SwitchoverEvent:
    timestamp: float  # wall clock
    old_primary: str
    new_primary: str
    duration_ms: float
    success: bool


class MetricsCollector:
    """Thread-safe collection of insert results and switchover events.
    Exposes Prometheus metrics and generates terminal reports."""

    def __init__(self, metrics_port: int = 9200):
        self._lock = threading.Lock()
        self._inserts: List[InsertResult] = []
        self._switchovers: List[SwitchoverEvent] = []
        self._test_start: float = 0.0
        self._metrics_port = metrics_port

        # Prometheus metrics
        self._prom_insert_latency = Histogram(
            "global_test_insert_latency_seconds",
            "Insert operation latency",
            buckets=[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0],
        )
        self._prom_insert_total = Counter(
            "global_test_insert_total",
            "Total insert operations",
            ["status"],
        )
        self._prom_switchover_total = Counter(
            "global_test_switchover_total",
            "Total switchover operations",
        )
        self._prom_switchover_duration = Histogram(
            "global_test_switchover_duration_seconds",
            "Switchover duration",
            buckets=[0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        )
        self._prom_current_primary = Gauge(
            "global_test_current_primary",
            "Current primary cluster (1=by-dev1, 2=by-dev2)",
        )

    def start(self):
        """Start Prometheus HTTP server and record test start time."""
        self._test_start = time.time()
        start_http_server(self._metrics_port)

    def record_insert(self, result: InsertResult):
        with self._lock:
            self._inserts.append(result)
        # Update Prometheus
        self._prom_insert_latency.observe(result.latency_ms / 1000.0)
        status = "success" if result.success else "fail"
        self._prom_insert_total.labels(status=status).inc()

    def record_switchover(self, event: SwitchoverEvent):
        with self._lock:
            self._switchovers.append(event)
        # Update Prometheus
        self._prom_switchover_total.inc()
        self._prom_switchover_duration.observe(event.duration_ms / 1000.0)
        primary_val = 1 if "19530" in event.new_primary else 2
        self._prom_current_primary.set(primary_val)

    def set_initial_primary(self, endpoint: str):
        primary_val = 1 if "19530" in endpoint else 2
        self._prom_current_primary.set(primary_val)

    def _in_switchover_window(self, ts: float, window_s: float) -> bool:
        """Check if a timestamp falls within any switchover window."""
        with self._lock:
            switchovers = list(self._switchovers)
        for ev in switchovers:
            if abs(ts - ev.timestamp) <= window_s:
                return True
        return False

    def _compute_percentile(self, values: List[float], p: float) -> float:
        if not values:
            return 0.0
        return float(np.percentile(values, p))

    def generate_report(self, switchover_window_s: float = 5.0) -> str:
        """Generate the terminal summary report."""
        with self._lock:
            inserts = list(self._inserts)
            switchovers = list(self._switchovers)

        duration = time.time() - self._test_start
        total = len(inserts)
        failures = sum(1 for r in inserts if not r.success)
        successes = total - failures
        failure_rate = failures / total if total > 0 else 0.0

        # Split latencies into steady-state vs switchover-window
        steady_latencies = []
        window_latencies = []
        for r in inserts:
            if r.success:
                if self._in_switchover_window(r.timestamp, switchover_window_s):
                    window_latencies.append(r.latency_ms)
                else:
                    steady_latencies.append(r.latency_ms)

        def fmt_lat(values, p):
            v = self._compute_percentile(values, p)
            return f"{v:.1f}" if values else "N/A"

        lines = [
            "",
            "=" * 64,
            " Global Client E2E Test Report",
            "=" * 64,
            f"Duration: {duration:.1f}s | Switchovers: {len(switchovers)} | "
            f"Strategy: 2 fast(2min) + 2 slow(4min)",
            f"Clusters: by-dev1 (localhost:19530) <-> by-dev2 (localhost:19531)",
            "",
            "--- Insert Summary ---",
            f"Total: {total} | Success: {successes} | Failed: {failures} | "
            f"Failure Rate: {failure_rate:.2%}",
            "",
            "--- Latency (ms) ---",
            f"{'':18s}{'Steady State':>14s}    {'Switchover Window':>20s}",
            f"  p50: {fmt_lat(steady_latencies, 50):>20s}    {fmt_lat(window_latencies, 50):>20s}",
            f"  p95: {fmt_lat(steady_latencies, 95):>20s}    {fmt_lat(window_latencies, 95):>20s}",
            f"  p99: {fmt_lat(steady_latencies, 99):>20s}    {fmt_lat(window_latencies, 99):>20s}",
            f"  max: {fmt_lat(steady_latencies, 100):>20s}    {fmt_lat(window_latencies, 100):>20s}",
            "",
            "--- Switchover Events ---",
        ]

        for i, ev in enumerate(switchovers):
            t_offset = ev.timestamp - self._test_start
            # Count failures in this switchover's window
            wf = sum(
                1
                for r in inserts
                if not r.success and abs(r.timestamp - ev.timestamp) <= switchover_window_s
            )
            lines.append(
                f"  #{i+1} [t={t_offset:.1f}s]  {ev.old_primary} -> {ev.new_primary}  "
                f"duration={ev.duration_ms/1000:.2f}s  window_failures={wf}"
            )

        lines.extend([
            "",
            "--- Prometheus Metrics ---",
            f"  Endpoint: http://localhost:{self._metrics_port}/metrics",
            "=" * 64,
            "",
        ])
        return "\n".join(lines)

    def get_results(self, switchover_window_s: float = 5.0) -> dict:
        """Return structured results for assertions."""
        with self._lock:
            inserts = list(self._inserts)
            switchovers = list(self._switchovers)

        total = len(inserts)
        failures = sum(1 for r in inserts if not r.success)
        failure_rate = failures / total if total > 0 else 0.0

        # Failures outside switchover windows
        outside_failures = sum(
            1
            for r in inserts
            if not r.success and not self._in_switchover_window(r.timestamp, switchover_window_s)
        )

        steady_latencies = [
            r.latency_ms
            for r in inserts
            if r.success and not self._in_switchover_window(r.timestamp, switchover_window_s)
        ]
        window_latencies = [
            r.latency_ms
            for r in inserts
            if r.success and self._in_switchover_window(r.timestamp, switchover_window_s)
        ]

        return {
            "total": total,
            "failures": failures,
            "failure_rate": failure_rate,
            "outside_failures": outside_failures,
            "switchover_count": len(switchovers),
            "all_switchovers_succeeded": all(ev.success for ev in switchovers),
            "steady_p50": self._compute_percentile(steady_latencies, 50),
            "steady_p99": self._compute_percentile(steady_latencies, 99),
            "window_p99": self._compute_percentile(window_latencies, 99),
        }
```

**Step 3: Verify file is importable**

Run: `cd /home/sheep/workspace/pymilvus && python -c "from tests.e2e_global_client.metrics_collector import MetricsCollector, InsertResult, SwitchoverEvent; print('OK')"`
Expected: `OK`

**Step 4: Commit**

```bash
git add tests/e2e_global_client/__init__.py tests/e2e_global_client/metrics_collector.py
git commit -m "feat(test): add metrics collector for global client E2E test

Signed-off-by: bigsheeper <yihao.dai@zilliz.com>"
```

---

### Task 2: Create topology_server.py — Mock REST Server + Switchover Orchestrator

**Files:**
- Create: `tests/e2e_global_client/topology_server.py`

**Reference:** `/home/sheep/workspace/snippets/tests/test_cdc/testcases/update_config.py` for switchover call order.

**Step 1: Write topology_server.py**

```python
# tests/e2e_global_client/topology_server.py
import json
import logging
import threading
import time
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

from pymilvus import MilvusClient

logger = logging.getLogger(__name__)

TOKEN = "root:Milvus"
PCHANNEL_NUM = 16


@dataclass
class ClusterConfig:
    cluster_id: str
    addr: str
    port: int

    @property
    def uri(self) -> str:
        return f"http://localhost:{self.port}"

    def pchannels(self) -> list:
        return [f"{self.cluster_id}-rootcoord-dml_{i}" for i in range(PCHANNEL_NUM)]


CLUSTER_A = ClusterConfig(cluster_id="by-dev1", addr="localhost", port=19530)
CLUSTER_B = ClusterConfig(cluster_id="by-dev2", addr="localhost", port=19531)


class TopologyState:
    """Thread-safe mutable topology state."""

    def __init__(self, initial_primary: ClusterConfig, initial_secondary: ClusterConfig):
        self._lock = threading.Lock()
        self._version = 1
        self._primary = initial_primary
        self._secondary = initial_secondary

    @property
    def version(self) -> int:
        with self._lock:
            return self._version

    @property
    def primary(self) -> ClusterConfig:
        with self._lock:
            return self._primary

    @property
    def secondary(self) -> ClusterConfig:
        with self._lock:
            return self._secondary

    def swap(self):
        """Swap primary and secondary, bump version."""
        with self._lock:
            self._primary, self._secondary = self._secondary, self._primary
            self._version += 1

    def topology_response(self) -> dict:
        """Build the REST API response."""
        with self._lock:
            return {
                "code": 0,
                "data": {
                    "version": str(self._version),
                    "clusters": [
                        {
                            "clusterId": self._primary.cluster_id,
                            "endpoint": self._primary.uri,
                            "capability": 3,  # PRIMARY
                        },
                        {
                            "clusterId": self._secondary.cluster_id,
                            "endpoint": self._secondary.uri,
                            "capability": 1,  # READABLE
                        },
                    ],
                },
            }


class TopologyRequestHandler(BaseHTTPRequestHandler):
    """HTTP handler that serves topology responses."""

    # Set by TopologyServer before starting
    topology_state: Optional[TopologyState] = None

    def do_GET(self):
        if self.path == "/global-cluster/topology":
            resp = self.topology_state.topology_response()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps(resp).encode())
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        logger.debug(f"TopologyServer: {format % args}")


class TopologyServer:
    """Mock topology REST server with switchover orchestration."""

    def __init__(
        self,
        server_port: int = 8080,
        primary: ClusterConfig = CLUSTER_A,
        secondary: ClusterConfig = CLUSTER_B,
    ):
        self.server_port = server_port
        self.state = TopologyState(primary, secondary)
        self._client_a: Optional[MilvusClient] = None
        self._client_b: Optional[MilvusClient] = None
        self._httpd: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    def start(self):
        """Start HTTP server and Milvus client connections."""
        # Connect to both clusters for switchover API calls
        self._client_a = MilvusClient(uri=CLUSTER_A.uri, token=TOKEN)
        self._client_b = MilvusClient(uri=CLUSTER_B.uri, token=TOKEN)

        # Start HTTP server
        handler = TopologyRequestHandler
        handler.topology_state = self.state
        self._httpd = HTTPServer(("0.0.0.0", self.server_port), handler)
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        logger.info(f"Topology server started on port {self.server_port}")

    def stop(self):
        """Stop HTTP server and close Milvus connections."""
        if self._httpd:
            self._httpd.shutdown()
            self._thread.join(timeout=5)
        if self._client_a:
            self._client_a.close()
        if self._client_b:
            self._client_b.close()
        logger.info("Topology server stopped")

    @property
    def url(self) -> str:
        return f"http://localhost:{self.server_port}"

    def _build_replicate_config(self, source: ClusterConfig, target: ClusterConfig) -> dict:
        """Build the update_replicate_configuration kwargs."""
        return {
            "clusters": [
                {
                    "cluster_id": CLUSTER_A.cluster_id,
                    "connection_param": {"uri": CLUSTER_A.uri, "token": TOKEN},
                    "pchannels": CLUSTER_A.pchannels(),
                },
                {
                    "cluster_id": CLUSTER_B.cluster_id,
                    "connection_param": {"uri": CLUSTER_B.uri, "token": TOKEN},
                    "pchannels": CLUSTER_B.pchannels(),
                },
            ],
            "cross_cluster_topology": [
                {
                    "source_cluster_id": source.cluster_id,
                    "target_cluster_id": target.cluster_id,
                }
            ],
        }

    def _get_client(self, cluster: ClusterConfig) -> MilvusClient:
        if cluster.cluster_id == CLUSTER_A.cluster_id:
            return self._client_a
        return self._client_b

    def init_replication(self):
        """Initialize CDC replication: current primary -> current secondary."""
        primary = self.state.primary
        secondary = self.state.secondary
        config = self._build_replicate_config(source=primary, target=secondary)

        # Old primary (secondary in this case doesn't exist yet) first, then primary
        self._get_client(secondary).update_replicate_configuration(**config)
        self._get_client(primary).update_replicate_configuration(**config)
        logger.info(
            f"CDC replication initialized: {primary.cluster_id} -> {secondary.cluster_id}"
        )

    def switchover(self) -> tuple:
        """Execute a switchover: demote old primary, promote new primary.

        Returns (old_primary_id, new_primary_id).
        """
        old_primary = self.state.primary
        new_primary = self.state.secondary

        # Build new config with swapped topology
        config = self._build_replicate_config(source=new_primary, target=old_primary)

        # Old primary first (demote), new primary second (promote)
        self._get_client(old_primary).update_replicate_configuration(**config)
        self._get_client(new_primary).update_replicate_configuration(**config)

        # Flip topology state
        self.state.swap()

        logger.info(
            f"Switchover complete: {old_primary.cluster_id} -> {new_primary.cluster_id}"
        )
        return old_primary.cluster_id, new_primary.cluster_id
```

**Step 2: Verify file is importable**

Run: `cd /home/sheep/workspace/pymilvus && python -c "from tests.e2e_global_client.topology_server import TopologyServer, CLUSTER_A, CLUSTER_B; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add tests/e2e_global_client/topology_server.py
git commit -m "feat(test): add mock topology server with switchover orchestration

Signed-off-by: bigsheeper <yihao.dai@zilliz.com>"
```

---

### Task 3: Create conftest.py — Pytest Fixtures and CLI Options

**Files:**
- Create: `tests/e2e_global_client/conftest.py`

**Key detail:** `is_global_endpoint` must be patched so that the mock server URL (which does NOT contain "global-cluster") is treated as a global endpoint. The patch target is `pymilvus.client.grpc_handler.is_global_endpoint` (where it's imported and used).

**Step 1: Write conftest.py**

```python
# tests/e2e_global_client/conftest.py
import logging
from unittest.mock import patch

import pytest

from pymilvus import MilvusClient
from tests.e2e_global_client.metrics_collector import MetricsCollector
from tests.e2e_global_client.topology_server import CLUSTER_A, CLUSTER_B, TopologyServer

logger = logging.getLogger(__name__)


def pytest_addoption(parser):
    parser.addoption("--switchover-interval", type=int, default=120,
                     help="Seconds between fast switchovers (default: 120)")
    parser.addoption("--test-duration", type=int, default=720,
                     help="Total test duration in seconds (default: 720)")
    parser.addoption("--max-failure-rate", type=float, default=0.05,
                     help="Max allowed insert failure rate (default: 0.05)")
    parser.addoption("--switchover-window", type=float, default=5.0,
                     help="Seconds around switchover to tolerate failures (default: 5)")
    parser.addoption("--primary-port", type=int, default=19530,
                     help="Initial primary Milvus port (default: 19530)")
    parser.addoption("--standby-port", type=int, default=19531,
                     help="Initial standby Milvus port (default: 19531)")
    parser.addoption("--metrics-port", type=int, default=9200,
                     help="Prometheus metrics port (default: 9200)")
    parser.addoption("--topo-server-port", type=int, default=8080,
                     help="Mock topology server port (default: 8080)")


@pytest.fixture(scope="session")
def test_config(request):
    return {
        "switchover_interval": request.config.getoption("--switchover-interval"),
        "test_duration": request.config.getoption("--test-duration"),
        "max_failure_rate": request.config.getoption("--max-failure-rate"),
        "switchover_window": request.config.getoption("--switchover-window"),
        "primary_port": request.config.getoption("--primary-port"),
        "standby_port": request.config.getoption("--standby-port"),
        "metrics_port": request.config.getoption("--metrics-port"),
        "topo_server_port": request.config.getoption("--topo-server-port"),
    }


@pytest.fixture(scope="session")
def metrics_collector(test_config):
    collector = MetricsCollector(metrics_port=test_config["metrics_port"])
    collector.start()
    return collector


@pytest.fixture(scope="session")
def topology_server(test_config):
    server = TopologyServer(server_port=test_config["topo_server_port"])
    server.start()
    yield server
    server.stop()


@pytest.fixture(scope="session")
def global_client(topology_server):
    """Create a MilvusClient that connects through the global client path."""
    mock_url = topology_server.url

    # Patch is_global_endpoint at the import site in grpc_handler
    # so GrpcHandler.__init__ takes the global path for our mock URL
    original_is_global = None

    def patched_is_global(uri):
        if uri and mock_url in uri:
            return True
        if original_is_global:
            return original_is_global(uri)
        return False

    with patch(
        "pymilvus.client.grpc_handler.is_global_endpoint",
        side_effect=patched_is_global,
    ):
        # Also patch in orm.connections where it's imported
        with patch(
            "pymilvus.orm.connections.is_global_endpoint",
            side_effect=patched_is_global,
        ):
            client = MilvusClient(uri=mock_url, token="root:Milvus")

    yield client
    client.close()
```

**Step 2: Verify fixture loading**

Run: `cd /home/sheep/workspace/pymilvus && python -c "from tests.e2e_global_client.conftest import pytest_addoption; print('OK')"`
Expected: `OK`

**Step 3: Commit**

```bash
git add tests/e2e_global_client/conftest.py
git commit -m "feat(test): add pytest fixtures for global client E2E test

Signed-off-by: bigsheeper <yihao.dai@zilliz.com>"
```

---

### Task 4: Create test_global_client_e2e.py — The E2E Test

**Files:**
- Create: `tests/e2e_global_client/test_global_client_e2e.py`

**Step 1: Write the test file**

```python
# tests/e2e_global_client/test_global_client_e2e.py
import logging
import random
import threading
import time

import pytest

from tests.e2e_global_client.metrics_collector import InsertResult, SwitchoverEvent

logger = logging.getLogger(__name__)

COLLECTION_NAME = "global_client_e2e_test"
DIM = 128


def _create_collection(client):
    """Create a test collection if it doesn't exist."""
    if client.has_collection(COLLECTION_NAME):
        client.drop_collection(COLLECTION_NAME)
    client.create_collection(
        collection_name=COLLECTION_NAME,
        dimension=DIM,
    )
    logger.info(f"Collection '{COLLECTION_NAME}' created")


def _generate_insert_data(batch_size: int = 10) -> list:
    """Generate a batch of random vectors for insertion."""
    return [
        {"id": int(time.time_ns()) + i, "vector": [random.random() for _ in range(DIM)]}
        for i in range(batch_size)
    ]


def _insert_loop(client, collector, stop_event: threading.Event):
    """Continuously insert data, recording latency and failures."""
    while not stop_event.is_set():
        data = _generate_insert_data(batch_size=10)
        start = time.perf_counter()
        ts = time.time()
        try:
            client.insert(collection_name=COLLECTION_NAME, data=data)
            latency_ms = (time.perf_counter() - start) * 1000.0
            collector.record_insert(InsertResult(
                timestamp=ts, latency_ms=latency_ms, success=True,
            ))
        except Exception as e:
            latency_ms = (time.perf_counter() - start) * 1000.0
            collector.record_insert(InsertResult(
                timestamp=ts, latency_ms=latency_ms, success=False, error=str(e),
            ))
            logger.warning(f"Insert failed: {e}")
        # Small sleep to avoid overwhelming
        time.sleep(0.05)


def _switchover_schedule(test_duration: int, fast_interval: int):
    """Generate switchover timestamps for 2 fast + 2 slow phases.

    Phase 1 (fast): 2 switchovers at fast_interval spacing.
    Phase 2 (slow): 2 switchovers evenly dividing remaining time.
    """
    times = []
    # Phase 1: 2 fast switchovers
    for i in range(2):
        times.append(fast_interval * i)

    # Phase 2: 2 slow switchovers over remaining time
    fast_end = fast_interval * 2
    remaining = test_duration - fast_end
    slow_interval = remaining // 2
    for i in range(2):
        times.append(fast_end + slow_interval * i)

    return times


class TestGlobalClientE2E:

    def test_switchover_resilience(
        self, global_client, topology_server, metrics_collector, test_config
    ):
        """Run continuous inserts while performing periodic switchovers.

        Phase 1 (fast): 2 switchovers every 2 minutes — tests error-triggered refresh.
        Phase 2 (slow): 2 switchovers every 4 minutes — tests background refresh.
        """
        duration = test_config["test_duration"]
        fast_interval = test_config["switchover_interval"]
        max_failure_rate = test_config["max_failure_rate"]
        window_s = test_config["switchover_window"]

        # Setup: init CDC replication and create collection
        topology_server.init_replication()
        _create_collection(global_client)
        metrics_collector.set_initial_primary(
            f"localhost:{test_config['primary_port']}"
        )

        # Schedule switchovers
        schedule = _switchover_schedule(duration, fast_interval)
        logger.info(f"Switchover schedule (offsets): {schedule}")

        # Start insert loop
        stop_event = threading.Event()
        insert_thread = threading.Thread(
            target=_insert_loop,
            args=(global_client, metrics_collector, stop_event),
            daemon=True,
        )
        insert_thread.start()
        logger.info("Insert loop started")

        # Execute switchovers on schedule
        test_start = time.time()
        for i, offset in enumerate(schedule):
            # Wait until scheduled time
            target_time = test_start + offset
            now = time.time()
            if target_time > now:
                time.sleep(target_time - now)

            logger.info(f"Executing switchover #{i+1} at t={time.time()-test_start:.1f}s")
            sw_start = time.perf_counter()
            sw_ts = time.time()
            try:
                old_id, new_id = topology_server.switchover()
                sw_duration_ms = (time.perf_counter() - sw_start) * 1000.0
                metrics_collector.record_switchover(SwitchoverEvent(
                    timestamp=sw_ts,
                    old_primary=old_id,
                    new_primary=new_id,
                    duration_ms=sw_duration_ms,
                    success=True,
                ))
                logger.info(
                    f"Switchover #{i+1} done: {old_id} -> {new_id} "
                    f"({sw_duration_ms:.0f}ms)"
                )
            except Exception as e:
                sw_duration_ms = (time.perf_counter() - sw_start) * 1000.0
                metrics_collector.record_switchover(SwitchoverEvent(
                    timestamp=sw_ts,
                    old_primary="unknown",
                    new_primary="unknown",
                    duration_ms=sw_duration_ms,
                    success=False,
                ))
                logger.error(f"Switchover #{i+1} failed: {e}")

        # Wait for remaining test duration
        elapsed = time.time() - test_start
        remaining = duration - elapsed
        if remaining > 0:
            logger.info(f"Waiting {remaining:.0f}s for remaining test duration...")
            time.sleep(remaining)

        # Stop insert loop
        stop_event.set()
        insert_thread.join(timeout=10)

        # Generate and print report
        report = metrics_collector.generate_report(switchover_window_s=window_s)
        print(report)

        # Cleanup
        try:
            global_client.drop_collection(COLLECTION_NAME)
        except Exception:
            pass

        # Assertions
        results = metrics_collector.get_results(switchover_window_s=window_s)

        # Hard assertions
        assert results["all_switchovers_succeeded"], "Not all switchovers succeeded"
        assert results["failure_rate"] <= max_failure_rate, (
            f"Failure rate {results['failure_rate']:.2%} exceeds max {max_failure_rate:.2%}"
        )
        assert results["outside_failures"] == 0, (
            f"{results['outside_failures']} failures occurred outside switchover windows"
        )

        # Soft assertions (warnings only)
        if results["window_p99"] > 5000:
            logger.warning(
                f"SOFT: p99 latency during switchover windows "
                f"({results['window_p99']:.0f}ms) exceeds 5000ms"
            )
        if results["steady_p50"] > 100:
            logger.warning(
                f"SOFT: p50 steady-state latency "
                f"({results['steady_p50']:.0f}ms) exceeds 100ms"
            )
```

**Step 2: Verify the test is collected by pytest (dry run)**

Run: `cd /home/sheep/workspace/pymilvus && python -m pytest tests/e2e_global_client/test_global_client_e2e.py --collect-only 2>&1 | head -20`
Expected: Shows `test_switchover_resilience` collected (may show import warnings if Milvus isn't running, but collection should work)

**Step 3: Commit**

```bash
git add tests/e2e_global_client/test_global_client_e2e.py
git commit -m "feat(test): add global client E2E switchover resilience test

Signed-off-by: bigsheeper <yihao.dai@zilliz.com>"
```

---

### Task 5: Verify Full Test (requires running Milvus)

**Prerequisites:** Two Milvus clusters running via `milvus_control -m start_milvus`, monitoring stack running.

**Step 1: Activate conda environment**

Run: `conda activate milvus2`

**Step 2: Install prometheus_client if needed**

Run: `pip install prometheus_client`

**Step 3: Run the E2E test**

Run: `cd /home/sheep/workspace/pymilvus && python -m pytest tests/e2e_global_client/test_global_client_e2e.py -v -s --test-duration=720 --switchover-interval=120 --log-cli-level=INFO 2>&1 | tail -80`

Expected: Test runs for 12 minutes, prints report, passes assertions.

**Step 4: Final commit (if any fixes needed)**

Fix any issues found during the live run and commit.
