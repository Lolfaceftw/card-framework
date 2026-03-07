from huggingface_hub import hf_hub_download
import pandas as pd
from openai import OpenAI
import json
from difflib import SequenceMatcher
import requests
from transformers import AutoTokenizer
from rich.console import Console
from rich.markdown import Markdown
from rich.live import Live
from unidecode import unidecode

# Set accordingly
VLLM_URL = "http://PRIVATE_ENDPOINT_REDACTED/v1"
VLLM_API_KEY = "EMPTY"
BIN_BOUNDARIES = [
    4096,
    8192,
    16384,
    32768,
    65536,
    131072,
    262144,
    524288,
    1048576,
]

tokenizer = AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="Qwen/Qwen3.5-35B-A3B"
)

dataset = pd.concat(
    [
        pd.read_parquet(
            hf_hub_download(
                repo_id="openai/mrcr",
                filename="8needle/8needle_0.parquet",
                repo_type="dataset",
            )
        ),
        pd.read_parquet(
            hf_hub_download(
                repo_id="openai/mrcr",
                filename="8needle/8needle_1.parquet",
                repo_type="dataset",
            )
        ),
    ]
)
console = Console()
rprint = console.print
client = OpenAI(base_url=VLLM_URL, api_key=VLLM_API_KEY)
scores: dict[str, list[float]] = {}


def grade(response: str, answer, random_string_to_prepend) -> float:
    """
    Compare response and answer.
    """
    if not response.strip().startswith(random_string_to_prepend):
        return 0
    response = response.strip().removeprefix(random_string_to_prepend)
    answer = answer.strip().removeprefix(random_string_to_prepend)
    return float(SequenceMatcher(None, response, answer).ratio())


def n_tokens(messages: list[dict]) -> int:
    """
    Count tokens in messages.
    """
    return sum(len(tokenizer.encode(m["content"])) for m in messages)


def qwen_tokenizer(messages: list[dict]) -> int:
    """
    Count tokens in messages using Qwen tokenizer.
    """
    return sum([len(tokenizer.encode(m["content"])) for m in messages])


def qwen_chat(messages: list[dict]) -> int:
    token_id = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=True
    )
    return len(token_id)


def resolve_bin(token_count: int) -> str:
    prev = 0
    for boundary in BIN_BOUNDARIES:
        if token_count <= boundary:
            if prev == 0:
                return f"(0, {boundary}]"
            return f"({prev}, {boundary}]"
        prev = boundary
    return None


response: dict = requests.get(f"{VLLM_URL}/models").json()
model: str = response["data"][0]["id"]
context_size: int = response["data"][0]["max_model_len"]

for index, row in dataset.iterrows():
    messages = json.loads(row["prompt"])
    rprint(
        f"[bold dark_orange3]{model}[/bold dark_orange3] [grey69]token count for idx[/grey69] [cyan]{index}[/cyan]: [cyan]{qwen_chat(messages=messages)}[/cyan]"
    )

    answer_tokens = len(tokenizer.encode(row["answer"]))
    total_tokens = qwen_chat(messages=messages) + answer_tokens

    if total_tokens > context_size - 10000:
        continue
    completion = client.chat.completions.create(
        model=model,
        messages=messages,
        stream=True,
        temperature=0.6,
        top_p=0.95,
        presence_penalty=0.0,
        extra_body={"chat_template_kwargs": {"enable_thinking": False},"top_k": 20, "min_p": 0.0, "repetition_penalty": 1.0},
    )
    reasoning = ""
    response = ""
    with Live(Markdown(""), vertical_overflow="crop_above") as live:
        # rprint("[dark_orange3]Thinking... [/dark_orange3]", end="")
        for chunk in completion:
            accum = ""
            if not chunk.choices:
                continue
            if (
                hasattr(chunk.choices[0].delta, "reasoning")
                and chunk.choices[0].delta.reasoning
            ):
                # rprint(chunk.choices[0].delta.reasoning, end="")
                reasoning += chunk.choices[0].delta.reasoning
            if chunk.choices[0].delta.content:
                rprint(chunk.choices[0].delta.content, end="")
                response += chunk.choices[0].delta.content
            live.update(Markdown((reasoning + response)[-500:]))
    response = unidecode(response)
    answer = unidecode(row["answer"])
    score = grade(response, answer, row["random_string_to_prepend"])
    rprint(f"[bold red3]Score:[/bold red3] [cyan]{score}[/cyan]")
    bin = resolve_bin(total_tokens)
    rprint(
        f"[grey69]Total tokens:[/grey69] [cyan]{total_tokens}[/cyan], [grey69]Answer tokens:[/grey69] [cyan]{answer_tokens}[/cyan], [grey69]Bin:[/grey69] [cyan]{bin}[/cyan], \n[grey69]Answer:[/grey69] [cyan]{repr(answer)}[/cyan], \n[grey69]Response:[/grey69] [cyan]{repr(response)}[/cyan]"
    )
    if bin is not None:
        if bin not in scores:
            scores[bin] = []
        scores[bin].append(score)

for bin, bin_scores in scores.items():
    rprint(
        f"[bold blue3]Bin:[/bold blue3] [cyan]{bin}[/cyan], [bold blue3]Average Score:[/bold blue3] [cyan]{sum(bin_scores) / len(bin_scores):.4f}[/cyan], [bold blue3]Count:[/bold blue3] [cyan]{len(bin_scores)}[/cyan]"
    )
