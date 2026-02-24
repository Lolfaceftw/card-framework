import json
import asyncio
from openai import OpenAI

# Mock tools definition
tools = [
    {
        "type": "function",
        "function": {
            "name": "test_tool",
            "description": "A test tool",
            "parameters": {"type": "object", "properties": {}},
        },
    }
]


async def main():
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key="sk-1e002d062181438ea18d1001fd49ff61",
    )
    model = "deepseek-reasoner"

    # DeepSeek docs state that for thinking mode + tool calls:
    # 1. We send user prompt
    # 2. Model returns reasoning_content + tool_calls + empty content
    # 3. We append the model's message exactly as is: role=assistant, tool_calls, reasoning_content, content
    # 4. We append the tool response: role=tool, tool_call_id, content
    # 5. We call the model again.
    # 6. When starting a NEW user question (turn 2), we should clear reasoning_content from history to save bandwidth.

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello! Use the test_tool."},
        # Model's first response (tool call)
        {
            "role": "assistant",
            "content": "",
            "reasoning_content": "I need to call the test tool.",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {"name": "test_tool", "arguments": "{}"},
                }
            ],
        },
        # Tool's response
        {
            "role": "tool",
            "tool_call_id": "call_1",
            "name": "test_tool",  # OpenAI spec uses name, some use just tool_call_id and content. Let's include name.
            "content": '{"result": "success"}',
        },
    ]

    print("--- MESSAGES TO SEND ---")
    print(json.dumps(messages, indent=2))

    try:
        response = client.chat.completions.create(
            model=model,
            messages=messages,
            # tools=tools, # Let's see if we even need to pass tools if we are just completing the flow
            stream=False,
        )
        print("Success! Response:")
        print(response.choices[0].message.content)
    except Exception as e:
        print(f"Error:")
        print(e)

        # In a2a, the LLM loop might generate an assistant message with NO tool calls,
        # and then a2a loop decides to ask it another question or force it to continue.
        # Let's test what happens if we send: User -> Assistant (text) -> Assistant (text).
        print("\n\n--- TESTING CONSECUTIVE ASSISTANT ---")
        bad_messages = [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
            {"role": "assistant", "content": "How are you?"},
        ]
        try:
            client.chat.completions.create(model=model, messages=bad_messages)
            print("Consecutive succeeded")
        except Exception as e2:
            print("Consecutive failed:", e2)


if __name__ == "__main__":
    asyncio.run(main())
