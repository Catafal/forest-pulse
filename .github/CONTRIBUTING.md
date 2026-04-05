# Contributing to Forest Pulse

Thanks for your interest in contributing!

## Getting Started

```bash
git clone https://github.com/jordicatafal/forest-pulse.git
cd forest-pulse
pip install -e ".[dev,train,notebooks]"
```

## Development Workflow

1. Create a branch: `git checkout -b feature/your-feature`
2. Make changes following the conventions in `CLAUDE.md`
3. Run tests: `pytest`
4. Run linter: `ruff check src/ tests/`
5. Open a PR with a clear description

## Code Conventions

- Max 200 lines per function, 1000 lines per file
- Type hints on all public functions
- Google-style docstrings
- Imports at top of file
- See `CLAUDE.md` for full conventions

## Reporting Issues

Please include:
- What you expected vs what happened
- Image type/source you were using (drone, ICGC, etc.)
- Python version and OS
- Full error traceback
