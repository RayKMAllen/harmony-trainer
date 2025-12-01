# ADR 0001: Project Bootstrap

## Status
Accepted

## Date
2024-12-01

## Context

Starting a new project for vocal harmony practice. The project needs to:
- Process audio files to separate vocals and extract pitch
- Provide real-time practice feedback
- Validate whether backing vocal extraction is viable (≥30-40% success rate)

The project was initialized from a one-pager document using the BootstrapFromOnePager mode.

## Decision

### Stack
- **Language:** Python 3.11+
- **Package Manager:** uv
- **Testing:** pytest
- **Linting:** ruff
- **Type Checking:** mypy
- **Pre-commit:** ruff hooks + hygiene

### Layout
- Package layout with `src/harmony_trainer/`
- Tests in `tests/`
- Documentation in `docs/`
- Helper scripts in `scripts/`

### Core Dependencies
- **Demucs:** Vocal/instrumental separation
- **svoice:** Lead/backing separation (experimental)
- **MedleyVox:** Lead/backing separation (alternative)
- **torchcrepe:** Pitch extraction
- **PyTorch:** ML runtime
- **Typer:** CLI framework
- **FastAPI:** Web UI backend

### Data Storage
- `data/songs.json` for song registry
- `data/stems/` for separated audio
- `data/analysis/` for pitch contours
- `eval/` for evaluation harness outputs

## Consequences

### Obligations
- Keep `docs/Architectural_Plan.md` updated with any structural changes
- Create ADRs for significant architectural decisions
- Maintain `project.todo.json` for task tracking
- Use TaskCycle/UpdatePlan modes for ongoing development

### Benefits
- Clear separation of concerns
- Standardized tooling and workflows
- Traceable development via task-driven approach

### Risks
- ML model dependencies may conflict
- GPU requirements for reasonable performance
- Backing extraction may not meet viability threshold

## Alternatives Considered

### Flat Layout (no src/)
Rejected: Package layout provides cleaner imports and better tool configuration.

### Poetry instead of uv
Rejected: uv is faster and simpler for this project's needs.

### Monorepo with multiple packages
Rejected: Overkill for a prototype; single package is sufficient.
