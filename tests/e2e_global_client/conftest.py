import logging
from unittest.mock import patch

import pytest

from pymilvus import MilvusClient
from tests.e2e_global_client.metrics_collector import MetricsCollector
from tests.e2e_global_client.topology_server import CLUSTER_A, CLUSTER_B, TopologyServer

logger = logging.getLogger(__name__)

# Shorter refresh interval for testing (default is 300s = 5min)
TEST_REFRESH_INTERVAL = 10  # seconds


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
    parser.addoption("--report-dir", type=str, default=None,
                     help="Directory to save test reports (default: None, stdout only)")
    parser.addoption("--keep-metrics-server", action="store_true", default=False,
                     help="Keep Prometheus metrics server running after test for Grafana scraping")


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
        "report_dir": request.config.getoption("--report-dir"),
        "keep_metrics_server": request.config.getoption("--keep-metrics-server"),
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

    def patched_is_global(uri):
        if uri and mock_url in uri:
            return True
        return False

    with patch(
        "pymilvus.client.grpc_handler.is_global_endpoint",
        side_effect=patched_is_global,
    ):
        with patch(
            "pymilvus.orm.connections.is_global_endpoint",
            side_effect=patched_is_global,
        ):
            client = MilvusClient(uri=mock_url, token="root:Milvus")

    # Shorten the topology refresh interval for testing.
    # The default is 300s (5min), which is too long for E2E tests.
    # We must stop and restart the refresher because the thread is already
    # sleeping in wait(300) — simply changing _refresh_interval won't wake it.
    handler = client._get_connection()
    if handler._global_stub and handler._global_stub._refresher:
        refresher = handler._global_stub._refresher
        refresher.stop()
        refresher._refresh_interval = TEST_REFRESH_INTERVAL
        refresher.start()
        logger.info(f"Restarted topology refresher with interval {TEST_REFRESH_INTERVAL}s")

    yield client
    client.close()
