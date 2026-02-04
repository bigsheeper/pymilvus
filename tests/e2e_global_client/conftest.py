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
