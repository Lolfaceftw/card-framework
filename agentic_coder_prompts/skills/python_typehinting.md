### Baseline
- Target Python `>=3.10`.
- Public functions, methods, class attributes, and module constants must be typed.
- All function signatures must include return types.
- Use modern syntax: `X | Y` (PEP 604), built-in generics (`list[str]`, `dict[str, int]`).
- Use `typing`/`typing_extensions` constructs when needed (`TypedDict`, `Protocol`, `Literal`, `TypeAlias`, `Self`).

### Design rules
- Prefer precise types over `Any`; justify unavoidable `Any` with a short comment.
- Model structured dict payloads with `TypedDict` or Pydantic models.
- Use `Protocol` for behavior-based interfaces.
- Use `dataclass(slots=True)` (or Pydantic models) for structured internal data where appropriate.
- Keep I/O boundaries explicit: parse/validate early, use typed domain objects internally.