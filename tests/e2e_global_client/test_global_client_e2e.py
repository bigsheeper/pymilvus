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

    Phase 1 (fast): 2 switchovers at fast_interval spacing, starting at fast_interval.
    Phase 2 (slow): 2 switchovers evenly dividing remaining time.
    """
    times = []
    # Phase 1: 2 fast switchovers (first one at fast_interval to allow steady-state inserts)
    for i in range(2):
        times.append(fast_interval * (i + 1))

    # Phase 2: 2 slow switchovers over remaining time
    fast_end = fast_interval * 3
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
        assert results["max_reconnect_ms"] <= 30000, (
            f"Client took {results['max_reconnect_ms']:.0f}ms to reconnect after switchover "
            f"(max allowed: 30000ms)"
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
