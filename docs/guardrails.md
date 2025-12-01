# Harmony Trainer - Development Guardrails

## Code Style & Tools

- **Language:** Python 3.11+
- **Package Manager:** uv
- **Testing:** pytest (integration-style tests preferred)
- **Linting:** ruff
- **Type Checking:** mypy
- **Pre-commit:** ruff + mypy + hygiene hooks

## Directory Boundaries

| Directory | Purpose |
|-----------|---------|
| `src/harmony_trainer/` | Application code |
| `tests/` | Test suite |
| `docs/` | Documentation, ADRs, bug reports |
| `scripts/` | Helper scripts |
| `data/` | Runtime data (songs, stems, analysis) |
| `eval/` | Evaluation harness outputs |

## Testing Expectations

- Prefer integration-style tests over heavily mocked unit tests
- Mocks allowed only for external systems (network, cloud, filesystem I/O)
- Every module should have at least one end-to-end test
- Focus on observable outcomes, not mocked interactions
- See `.roo/rules/02-no_mock.txt` for details

## Git & Commit Practices

- Use conventional commit style: `type(scope): description`
- Types: feat, fix, docs, chore, refactor, test
- Keep commits atomic and focused
- Reference task IDs from `project.todo.json` in commit messages

## Security & Secrets

- No secrets in code or logs
- Use environment variables for sensitive configuration
- Never commit API keys, tokens, or credentials

## Development Environment

- **Platform:** Windows (CMD/PowerShell)
- **Command chaining:** Use semicolons (`;`), not `&&`
- All shell commands must be Windows-compatible
- Use `uv run` prefix for all Python tool invocations

## Working Agreements

- All work must trace to a task in `project.todo.json`
- Architectural changes require ADR updates first
- Definition of done must be met before marking tasks complete
- Concise outputs preferred (see `.roo/rules/01-general.txt`)
