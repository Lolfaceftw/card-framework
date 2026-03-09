from typing import Any

from jinja2 import Environment, FileSystemLoader, select_autoescape

from card_framework.shared.paths import PROMPTS_DIR


class PromptManager:
    """Central registry for all prompts used across agents using Jinja2 templates."""

    _instance = None
    _env = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(PromptManager, cls).__new__(cls)
            cls._env = Environment(
                loader=FileSystemLoader(str(PROMPTS_DIR)),
                autoescape=select_autoescape(),
                trim_blocks=True,
                lstrip_blocks=True,
            )
        return cls._instance

    @classmethod
    def get_prompt(cls, key: str, **kwargs: Any) -> str:
        """Loads and renders a Jinja2 template by key (filename without extension)."""
        # Ensure the environment is initialized
        if cls._env is None:
            cls()

        template_file = f"{key}.jinja2"
        try:
            template = cls._env.get_template(template_file)
            return template.render(**kwargs)
        except Exception as e:
            raise KeyError(
                f"Prompt template '{template_file}' not found or could not be rendered: {e}"
            )
