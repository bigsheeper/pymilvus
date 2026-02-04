import threading
import time
from dataclasses import dataclass
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

        # Compute max reconnect time: for each switchover, find the first
        # successful insert after it, and measure the gap.
        max_reconnect_ms = 0.0
        for ev in switchovers:
            first_success_after = None
            for r in inserts:
                if r.success and r.timestamp > ev.timestamp:
                    first_success_after = r
                    break
            if first_success_after is not None:
                reconnect_ms = (first_success_after.timestamp - ev.timestamp) * 1000.0
                max_reconnect_ms = max(max_reconnect_ms, reconnect_ms)

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
            "max_reconnect_ms": max_reconnect_ms,
        }
