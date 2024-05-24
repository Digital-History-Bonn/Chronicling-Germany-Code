"""This module implements our CI function calls."""
import nox


@nox.session(name="test")
def run_test(session):
    """Run pytest."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest")


@nox.session(name="fast-test")
def run_test_fast(session):
    """Run pytest."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.run("pytest", "-m", "not slow")


@nox.session(name="lint")
def lint(session):
    """Check code conventions."""
    session.install("-r", "requirements.txt")
    session.install("pylint")
    session.run("pylint", "src", "tests", "script")


@nox.session(name="typing")
def mypy(session):
    """Check type hints."""
    session.install("-r", "requirements.txt")
    session.install("mypy")
    session.run(
        "mypy",
        "--install-types",
        "--non-interactive",
        "--ignore-missing-imports",
        "--strict",
        "--no-warn-return-any",
        "--explicit-package-bases",
        "--namespace-packages",
        "--implicit-reexport",  # tensorboard is untyped
        "--allow-untyped-calls",  # tensorboard is untyped
        "src/baseline_detection",
    )


@nox.session(name="format")
def format(session):
    """Fix common convention problems automatically."""
    session.install("black")
    session.install("isort")
    session.run("isort", "src", "script", "tests", "noxfile.py")
    session.run("black", "src", "script", "tests", "noxfile.py")


@nox.session(name="coverage")
def check_coverage(session):
    """Check test coverage and generate a html report."""
    session.install("-r", "requirements.txt")
    session.install("pytest")
    session.install("coverage")
    try:
        session.run("coverage", "run", "-m", "pytest")
    finally:
        session.run("coverage", "html")


@nox.session(name="coverage-clean")
def clean_coverage(session):
    """Remove the code coverage website."""
    session.run("rm", "-r", "htmlcov", external=True)
