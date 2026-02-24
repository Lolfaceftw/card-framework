"""
Logging Decorator for LLM Providers
====================================
Wraps an existing LLMProvider to capture its inputs and outputs.
"""

import json

from llm_provider import LLMProvider
from logger_utils import logger


class LoggingLLMProvider(LLMProvider):
    """
    Decorator for LLMProvider that logs all generation requests and responses.
    """

    def __init__(self, inner_provider: LLMProvider) -> None:
        self.inner_provider = inner_provider
        logger.info(
            f"LoggingLLMProvider initialized wrapping {type(inner_provider).__name__}"
        )

    def generate(
        self,
        system_prompt: str,
        user_prompt: str,
        max_tokens: int | None = None,
    ) -> str:
        logger.info("-" * 40)
        logger.info(f"LLM REQUEST - {type(self.inner_provider).__name__}")
        logger.info(f"System Prompt:\n{system_prompt}")
        logger.info(f"User Prompt:\n{user_prompt}")
        if max_tokens:
            logger.info(f"Max Tokens: {max_tokens}")

        try:
            response = self.inner_provider.generate(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_tokens,
            )
            logger.info(f"LLM RESPONSE:\n{response}")
            logger.info("-" * 40)
            return response
        except Exception as e:
            logger.error(f"LLM ERROR: {str(e)}")
            logger.info("-" * 40)
            raise

    def chat(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        tool_choice: str | dict | None = None,
        max_tokens: int | None = None,
    ):
        logger.info("-" * 40)
        logger.info(f"LLM CHAT REQUEST - {type(self.inner_provider).__name__}")
        try:
            # Use a custom default to handle non-serializable objects (like ChatCompletionMessage)
            def json_default(obj):
                if hasattr(obj, "model_dump"):
                    return obj.model_dump()
                return str(obj)

            logger.info(
                f"Messages: {json.dumps(messages, indent=2, default=json_default)}"
            )
        except Exception as e:
            logger.warning(f"Failed to log messages: {e}")

        if tools:
            logger.info(f"Tools: {json.dumps(tools, indent=2)}")
        if tool_choice:
            logger.info(f"Tool Choice: {tool_choice}")

        try:
            response_message = self.inner_provider.chat(
                messages=messages,
                tools=tools,
                tool_choice=tool_choice,
                max_tokens=max_tokens,
            )
            # Log the response content and tool calls if any
            res_log = {
                "role": response_message.role,
                "content": response_message.content,
            }
            if hasattr(response_message, "tool_calls") and response_message.tool_calls:
                res_log["tool_calls"] = []
                for tc in response_message.tool_calls:
                    if isinstance(tc, dict):
                        res_log["tool_calls"].append(
                            {
                                "id": tc.get("id"),
                                "function": {
                                    "name": tc.get("function", {}).get("name"),
                                    "arguments": tc.get("function", {}).get(
                                        "arguments"
                                    ),
                                },
                            }
                        )
                    else:
                        res_log["tool_calls"].append(
                            {
                                "id": tc.id,
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                        )
            logger.info(f"LLM CHAT RESPONSE:\n{json.dumps(res_log, indent=2)}")
            logger.info("-" * 40)
            return response_message
        except Exception as e:
            logger.error(f"LLM CHAT ERROR: {str(e)}")
            logger.info("-" * 40)
            raise
