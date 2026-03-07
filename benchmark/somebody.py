from openai import OpenAI
from test_ui import App
import requests

VLLM_URL = "http://202.92.159.240:8001/v1"
VLLM_API_KEY = "EMPTY"

client = OpenAI(
    base_url=VLLM_URL,
    api_key=VLLM_API_KEY
)


if __name__=="__main__":
    ui = App()
    ui.run()
    try:
        response = requests.get(f"{VLLM_URL}/models").json()
        print(response)
    except Exception as e:
        print(f"Error occurred: {e}")
    