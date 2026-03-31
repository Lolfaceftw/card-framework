"""Manual Textual browser for inspecting a local vLLM model endpoint."""

from __future__ import annotations

from textual import on
from textual.app import App, ComposeResult
from textual.widgets import Label, ListItem, ListView, RichLog
import requests

DEFAULT_VLLM_URL = "http://127.0.0.1:8001/v1"


class VllmModelBrowserApp(App[None]):
    """Simple manual UI for probing a local vLLM server."""

    def compose(self) -> ComposeResult:
        yield ListView(
            ListItem(Label("Show models", id="show-models")),
            ListItem(Label("Show endpoint", id="show-endpoint")),
            ListItem(Label("Exit", id="exit-app")),
        )
        yield RichLog(highlight=True, wrap=True)

    @on(ListView.Selected, "#show-models")
    def show_models(self) -> None:
        """Fetch and render the available vLLM models."""
        log = self.query_one(RichLog)
        try:
            response = requests.get(f"{DEFAULT_VLLM_URL}/models", timeout=10)
            response.raise_for_status()
        except requests.RequestException as exc:
            log.write(f"Request failed: {exc}")
            return
        log.write(response.json())

    @on(ListView.Selected, "#show-endpoint")
    def show_endpoint(self) -> None:
        """Show the currently configured endpoint."""
        self.query_one(RichLog).write({"vllm_url": DEFAULT_VLLM_URL})

    @on(ListView.Selected, "#exit-app")
    def exit_app(self) -> None:
        """Close the manual UI."""
        self.exit()


def main() -> None:
    """Launch the manual vLLM model browser."""
    VllmModelBrowserApp().run()


if __name__ == "__main__":
    main()
