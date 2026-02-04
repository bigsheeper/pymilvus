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
