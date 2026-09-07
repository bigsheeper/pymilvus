"""Idempotency key propagation: CallContext -> gRPC metadata, and the client entry points."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import pytest_asyncio
from pymilvus import AsyncMilvusClient, DataType, MilvusClient, connections
from pymilvus.client.call_context import CallContext
from pymilvus.client.connection_manager import AsyncConnectionManager, ConnectionManager

IDEMPOTENCY_KEY_HEADER = "idempotency-key"
_SCHEMA = {"fields": [{"name": "id", "is_primary": True, "type": DataType.INT64}]}


def _metadata_values(context: CallContext, key: str):
    return [v for k, v in context.to_grpc_metadata() if k == key]


def _context_of(mock_call):
    _, kwargs = mock_call.call_args
    return kwargs["context"]


@pytest.fixture(autouse=True)
def _reset_connection_managers():
    ConnectionManager._reset_instance()
    AsyncConnectionManager._reset_instance()
    yield
    ConnectionManager._reset_instance()
    AsyncConnectionManager._reset_instance()


class TestCallContext:
    def test_metadata_carries_idempotency_key(self):
        ctx = CallContext(db_name="db", idempotency_key="order-4711")
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == ["order-4711"]

    def test_metadata_omits_absent_key(self):
        ctx = CallContext(db_name="db")
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == []

    def test_metadata_omits_empty_key(self):
        ctx = CallContext(db_name="db", idempotency_key="")
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == []


def _sync_handler():
    handler = MagicMock()
    handler.get_server_type.return_value = "milvus"
    handler._wait_for_channel_ready = MagicMock()
    handler._get_schema.return_value = (_SCHEMA, 100)
    return handler


class TestMilvusClient:
    def test_insert_forwards_idempotency_key(self):
        handler = _sync_handler()
        handler.insert_rows.return_value = MagicMock(insert_count=1, primary_keys=[1], cost=0)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            MilvusClient().insert("col", {"id": 1}, idempotency_key="order-4711")

        ctx = _context_of(handler.insert_rows)
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == ["order-4711"]

    def test_insert_without_key_sends_no_header(self):
        handler = _sync_handler()
        handler.insert_rows.return_value = MagicMock(insert_count=1, primary_keys=[1], cost=0)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            MilvusClient().insert("col", {"id": 1})

        ctx = _context_of(handler.insert_rows)
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == []

    def test_upsert_forwards_idempotency_key(self):
        handler = _sync_handler()
        handler.upsert_rows.return_value = MagicMock(upsert_count=1, primary_keys=[1], cost=0)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            MilvusClient().upsert("col", {"id": 1}, idempotency_key="order-4711")

        ctx = _context_of(handler.upsert_rows)
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == ["order-4711"]

    def test_delete_forwards_idempotency_key(self):
        handler = _sync_handler()
        handler.delete.return_value = MagicMock(delete_count=1, primary_keys=[], cost=0)
        with patch("pymilvus.client.grpc_handler.GrpcHandler", return_value=handler):
            MilvusClient().delete("col", ids=[1], idempotency_key="order-4711")

        ctx = _context_of(handler.delete)
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == ["order-4711"]


@pytest_asyncio.fixture
async def async_client_and_handler():
    handler = MagicMock()
    handler.ensure_channel_ready = AsyncMock()
    handler._get_schema = AsyncMock(return_value=(_SCHEMA, 100))
    with patch("pymilvus.client.async_grpc_handler.AsyncGrpcHandler", return_value=handler):
        client = AsyncMilvusClient()
        await client._connect()
        yield client, handler


class TestAsyncMilvusClient:
    @pytest.mark.asyncio
    async def test_insert_forwards_idempotency_key(self, async_client_and_handler):
        client, handler = async_client_and_handler
        handler.insert_rows = AsyncMock(
            return_value=MagicMock(insert_count=1, primary_keys=[1], cost=0)
        )

        await client.insert("col", {"id": 1}, idempotency_key="order-4711")

        ctx = _context_of(handler.insert_rows)
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == ["order-4711"]

    @pytest.mark.asyncio
    async def test_upsert_forwards_idempotency_key(self, async_client_and_handler):
        client, handler = async_client_and_handler
        handler.upsert_rows = AsyncMock(
            return_value=MagicMock(upsert_count=1, primary_keys=[1], cost=0)
        )

        await client.upsert("col", {"id": 1}, idempotency_key="order-4711")

        ctx = _context_of(handler.upsert_rows)
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == ["order-4711"]

    @pytest.mark.asyncio
    async def test_delete_forwards_idempotency_key(self, async_client_and_handler):
        client, handler = async_client_and_handler
        handler.delete = AsyncMock(return_value=MagicMock(delete_count=1, primary_keys=[], cost=0))

        await client.delete("col", ids=[1], idempotency_key="order-4711")

        ctx = _context_of(handler.delete)
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == ["order-4711"]


class TestOrmConnections:
    def test_generate_call_context_carries_idempotency_key(self):
        ctx = connections._generate_call_context("unknown_alias", idempotency_key="order-4711")
        assert _metadata_values(ctx, IDEMPOTENCY_KEY_HEADER) == ["order-4711"]
