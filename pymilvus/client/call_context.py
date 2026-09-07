from typing import Optional

from pymilvus.client.utils import current_time_ms


def _api_level_md(context: Optional["CallContext"]) -> Optional[list]:
    if context is None:
        return None
    return context.to_grpc_metadata()


class CallContext:
    def __init__(self, db_name: str = "", client_request_id: str = "", idempotency_key: str = ""):
        self._db_name = db_name
        self._client_request_id = client_request_id
        self._idempotency_key = idempotency_key

    def to_grpc_metadata(self):
        metadata = [
            ("dbname", self._db_name),
            ("client-request-id", self._client_request_id),
            ("client-request-unixmsec", current_time_ms()),
        ]
        # Only sent when set: the server treats an empty metadata value as a
        # present-but-empty key, which is not the same as no key.
        if self._idempotency_key:
            metadata.append(("idempotency-key", self._idempotency_key))
        return metadata

    def get_db_name(self):
        return self._db_name
