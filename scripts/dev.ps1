# Development helper script for Windows PowerShell
# Usage: .\scripts\dev.ps1 [command]

param(
    [Parameter(Position=0)]
    [string]$Command = "help"
)

switch ($Command) {
    "sync" {
        Write-Host "Installing dependencies..."
        uv sync --all-extras
    }
    "test" {
        Write-Host "Running tests..."
        uv run pytest
    }
    "lint" {
        Write-Host "Running linter..."
        uv run ruff check src tests
    }
    "format" {
        Write-Host "Formatting code..."
        uv run ruff format src tests
    }
    "typecheck" {
        Write-Host "Running type checker..."
        uv run mypy src
    }
    "check" {
        Write-Host "Running all checks..."
        uv run ruff check src tests
        uv run ruff format --check src tests
        uv run mypy src
        uv run pytest
    }
    "precommit" {
        Write-Host "Installing pre-commit hooks..."
        uv run pre-commit install
    }
    default {
        Write-Host "Harmony Trainer Development Commands:"
        Write-Host "  sync      - Install dependencies"
        Write-Host "  test      - Run tests"
        Write-Host "  lint      - Run linter"
        Write-Host "  format    - Format code"
        Write-Host "  typecheck - Run type checker"
        Write-Host "  check     - Run all checks"
        Write-Host "  precommit - Install pre-commit hooks"
    }
}
