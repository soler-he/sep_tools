# noxfile.py
import nox

nox.options.sessions = ["lint", "tests"]
nox.options.default_venv_backend = "uv"

PYTHON_VERSIONS = ["3.10", "3.11", "3.12", "3.13", "3.14"]
GROUPS = [1, 2, 3, 4, 5]
SPLITS = 5

DEPENDENCIES = [
    "nbmake", "pytest", "pytest-doctestplus", "pytest-cov",
    "pytest-mpl", "pytest-xdist", "pytest-split",
]

COMMON_PYTEST_ARGS = [
    "-ra",
    "--mpl",
    "--mpl-baseline-path=baseline",
    "--mpl-baseline-relative",
    "--mpl-generate-summary=html",
    "--nbmake",
    "-n=auto",
    "--nbmake-kernel=python3",
    "--nbmake-timeout=500",
    "--ignore=SEP_Multi-Instrument-Plot.ipynb",
    "--ignore=SEP_PyOnset.ipynb",
    "--durations=0",
    "--store-durations",
]


@nox.session(venv_backend="uv")
def lint(session: nox.Session) -> None:
    """Run ruff linting."""
    session.install("ruff")
    session.run("ruff", "check", ".", "--select=E9,F63,F7,F82",
                "--exclude=.venv")
    session.run("ruff", "check", ".", "--select=E,F,C90", "--ignore=E501",
                "--exclude=.venv", "--statistics", "--quiet", "--exit-zero")
    session.run("ruff", "check", ".", "--select=E,F,C90", "--ignore=E501",
                "--exclude=.venv", "--exit-zero")


@nox.session(python=PYTHON_VERSIONS, venv_backend="uv")
def tests(session: nox.Session) -> None:
    """Run full test suite locally without splitting."""
    session.install(*DEPENDENCIES)
    session.install("-r", "requirements.txt")
    session.run(
        "pytest",
        *COMMON_PYTEST_ARGS,
        f"--mpl-results-path=tests_report/{session.python}",
        f"--cov-report=xml:coverage-{session.python}.xml",
    )


@nox.session(python=PYTHON_VERSIONS, venv_backend="uv")
@nox.parametrize("group", GROUPS)
def tests_split(session: nox.Session, group: int) -> None:
    """Mirror CI exactly — use to reproduce a specific CI group failure."""
    session.install(*DEPENDENCIES)
    session.install("-r", "requirements.txt")
    session.run(
        "pytest",
        *COMMON_PYTEST_ARGS,
        f"--mpl-results-path=tests_report/{session.python}-group{group}",
        f"--cov-report=xml:coverage-{session.python}-group{group}.xml",
        f"--splits={SPLITS}",
        f"--group={group}",
        "--splitting-algorithm=least_duration",
    )
