import asyncio
import json
import uuid
import threading
import time

import httpx

from events import event_bus


class AgentClient:
    """
    Robust HTTP client for communicating with A2A agents via JSON-RPC.
    Features connection pooling and exponential backoff for transient errors.
    """

    def __init__(self):
        self.limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        self._clients = {}
        self._lock = threading.Lock()

    def _get_client(self) -> httpx.AsyncClient:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        with self._lock:
            # Prune closed loops to prevent memory leaks if many run_until_complete cycles happen
            closed_loops = [
                loop_obj
                for loop_obj in self._clients
                if loop_obj is not None and loop_obj.is_closed()
            ]
            for loop_obj in closed_loops:
                self._clients.pop(loop_obj, None)

            client = self._clients.get(loop)
            if client is None or client.is_closed:
                client = httpx.AsyncClient(limits=self.limits)
                self._clients[loop] = client
            return client

    async def close(self):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        with self._lock:
            client = self._clients.pop(loop, None)

        if client is not None:
            await client.aclose()

    async def send_task(
        self,
        port: int,
        task_data: dict | str | object,
        timeout: float = 120.0,
        max_retries: int = 3,
        metadata: dict | None = None,
    ) -> str:
        """
        Sends a task to the agent.
        """
        call_id = str(uuid.uuid4())
        metadata = metadata or {}

        if hasattr(task_data, "model_dump_json"):
            task_text = task_data.model_dump_json()
        elif hasattr(task_data, "dict"):
            task_text = json.dumps(task_data.dict())
        elif isinstance(task_data, dict):
            task_text = json.dumps(task_data)
        else:
            task_text = str(task_data)

        payload = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "message/send",
            "params": {
                "message": {
                    "role": "user",
                    "parts": [{"kind": "text", "text": task_text}],
                    "messageId": str(uuid.uuid4()),
                }
            },
        }
        url = f"http://127.0.0.1:{port}"

        event_bus.publish(
            "a2a_call_started",
            call_id=call_id,
            port=port,
            timeout=timeout,
            max_retries=max_retries,
            **metadata,
        )

        last_exception = None
        for attempt in range(1, max_retries + 1):
            attempt_start = time.perf_counter()
            try:
                client = self._get_client()
                resp = await client.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                data = resp.json()

                if "error" in data:
                    raise RuntimeError(f"A2A error: {data['error']}")

                result = data.get("result", {})

                # Extract text
                for key in ["artifacts", "parts"]:
                    if key in result:
                        items = result[key]
                        texts = (
                            [
                                p["text"]
                                for i in items
                                for p in i.get("parts", [])
                                if p.get("kind") == "text"
                            ]
                            if key == "artifacts"
                            else [p["text"] for p in items if p.get("kind") == "text"]
                        )
                        if texts:
                            latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                            event_bus.publish(
                                "a2a_call_succeeded",
                                call_id=call_id,
                                port=port,
                                attempt=attempt,
                                latency_ms=latency_ms,
                                **metadata,
                            )
                            return "\n".join(texts)

                if "result" in result and "parts" in result["result"]:
                    texts = [
                        p["text"]
                        for p in result["result"]["parts"]
                        if p.get("kind") == "text"
                    ]
                    if texts:
                        latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                        event_bus.publish(
                            "a2a_call_succeeded",
                            call_id=call_id,
                            port=port,
                            attempt=attempt,
                            latency_ms=latency_ms,
                            **metadata,
                        )
                        return "\n".join(texts)

                latency_ms = int((time.perf_counter() - attempt_start) * 1000)
                event_bus.publish(
                    "a2a_call_succeeded",
                    call_id=call_id,
                    port=port,
                    attempt=attempt,
                    latency_ms=latency_ms,
                    **metadata,
                )
                return str(result)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                if attempt < max_retries:
                    delay_seconds = float(2**attempt)
                    event_bus.publish(
                        "a2a_call_retry",
                        call_id=call_id,
                        port=port,
                        attempt=attempt,
                        delay_seconds=delay_seconds,
                        error=str(e),
                        **metadata,
                    )
                    await asyncio.sleep(delay_seconds)  # Exponential backoff (2s, 4s, 8s)

        error_message = f"Failed after {max_retries} attempts. Last error: {last_exception}"
        event_bus.publish(
            "a2a_call_failed",
            call_id=call_id,
            port=port,
            error=error_message,
            **metadata,
        )
        raise RuntimeError(error_message)


# Global connection pool instance
agent_client = AgentClient()
