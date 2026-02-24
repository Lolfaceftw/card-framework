import uuid

import httpx


async def a2a_send_task(port: int, task_text: str, timeout: float = 120.0) -> str:
    """Send a JSON-RPC message/send to a local A2A server and return the text result."""
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
    async with httpx.AsyncClient(timeout=timeout) as client:
        resp = await client.post(url, json=payload)
        resp.raise_for_status()

    data = resp.json()

    # Handle JSON-RPC errors
    if "error" in data:
        raise RuntimeError(f"A2A error: {data['error']}")

    result = data.get("result", {})

    # Extract text from various A2A response shapes
    if "artifacts" in result:
        texts = []
        for artifact in result["artifacts"]:
            for part in artifact.get("parts", []):
                if part.get("kind") == "text":
                    texts.append(part["text"])
        if texts:
            return "\n".join(texts)

    if "parts" in result:
        texts = []
        for part in result["parts"]:
            if part.get("kind") == "text":
                texts.append(part["text"])
        if texts:
            return "\n".join(texts)

    if "result" in result:
        nested = result["result"]
        if "parts" in nested:
            texts = []
            for part in nested["parts"]:
                if part.get("kind") == "text":
                    texts.append(part["text"])
            if texts:
                return "\n".join(texts)

    return str(result)
