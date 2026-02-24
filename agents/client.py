import asyncio
import json
import uuid

import httpx


class AgentClient:
    """
    Robust HTTP client for communicating with A2A agents via JSON-RPC.
    Features connection pooling and exponential backoff for transient errors.
    """

    def __init__(self):
        self.limits = httpx.Limits(max_keepalive_connections=20, max_connections=100)
        self._client = None

    def _get_client(self) -> httpx.AsyncClient:
        if self._client is None or self._client.is_closed:
            self._client = httpx.AsyncClient(limits=self.limits)
        return self._client

    async def close(self):
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    async def send_task(
        self,
        port: int,
        task_data: dict | str | object,
        timeout: float = 120.0,
        max_retries: int = 3,
    ) -> str:
        """
        Sends a task to the agent.
        """
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

        last_exception = None
        for attempt in range(1, max_retries + 1):
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
                            return "\n".join(texts)

                if "result" in result and "parts" in result["result"]:
                    texts = [
                        p["text"]
                        for p in result["result"]["parts"]
                        if p.get("kind") == "text"
                    ]
                    if texts:
                        return "\n".join(texts)

                return str(result)
            except (httpx.RequestError, httpx.HTTPStatusError) as e:
                last_exception = e
                if attempt < max_retries:
                    await asyncio.sleep(2**attempt)  # Exponential backoff (2s, 4s, 8s)

        raise RuntimeError(
            f"Failed after {max_retries} attempts. Last error: {last_exception}"
        )


# Global connection pool instance
agent_client = AgentClient()
