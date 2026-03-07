from textual.app import App, ComposeResult
from textual.widgets import RichLog, ListView, ListItem, Label
from textual import on
import requests
        
class App(App):
    def compose(self) -> ComposeResult:
        yield ListView(
            ListItem(Label("Run", id="run")),
            ListItem(Label("Config", id="config")),
            ListItem(Label("Exit", id="exit"))
        )
        yield RichLog(highlight=True, wrap=True)
    @on(ListView.Selected, "#run")
    def run(self) -> None:
        text = self.query_one(RichLog)
        text.write(requests.get("http://202.92.159.240:8001/v1/models").json())

    @on(ListView.Selected, "#exit")
    def exit(self) -> None:
        self.exit()