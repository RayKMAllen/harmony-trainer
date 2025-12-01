"""Tests for the CLI entry point."""

from typer.testing import CliRunner

from harmony_trainer.cli import app

runner = CliRunner()


def test_cli_help() -> None:
    """Test that the CLI help command works."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "Harmony Trainer CLI" in result.stdout
    assert "Usage" in result.stdout


def test_cli_version() -> None:
    """Test that we can check the version via a hypothetical command or just help."""
    # Since we haven't implemented a version command yet, help is enough for now.
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
