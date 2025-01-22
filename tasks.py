import os

from invoke import Context, task

WINDOWS = os.name == "nt"
PROJECT_NAME = "fruit_vegetable_classification"
PYTHON_VERSION = "3.11"

# Setup commands
@task
def create_environment(ctx: Context) -> None:
    """Create a new conda environment for project."""
    ctx.run(
        f"conda create --name {PROJECT_NAME} python={PYTHON_VERSION} pip --no-default-packages --yes",
        echo=True,
        pty=not WINDOWS,
    )

@task
def requirements(ctx: Context) -> None:
    """Install project requirements."""
    ctx.run("pip install -U pip setuptools wheel", echo=True, pty=not WINDOWS)
    ctx.run("pip install -r requirements.txt", echo=True, pty=not WINDOWS)
    ctx.run("pip install -e .", echo=True, pty=not WINDOWS)


@task(requirements)
def dev_requirements(ctx: Context) -> None:
    """Install development requirements."""
    ctx.run('pip install -e .["dev"]', echo=True, pty=not WINDOWS)

# Project commands
@task
def preprocess_data(ctx: Context) -> None:
    """Preprocess data."""
    ctx.run(f"python src/{PROJECT_NAME}/data.py", echo=True, pty=not WINDOWS)

@task
def train(ctx: Context) -> None:
    """Train model."""
    ctx.run(f"python src/{PROJECT_NAME}/train.py", echo=True, pty=not WINDOWS)

@task
def test_model(ctx: Context) -> None:
    """Test model."""
    ctx.run(f"python src/{PROJECT_NAME}/test.py", echo=True, pty=not WINDOWS)

@task
def test(ctx: Context) -> None:
    """Run tests."""
    ctx.run("coverage run -m pytest tests/", echo=True, pty=not WINDOWS)
    ctx.run("coverage report -m", echo=True, pty=not WINDOWS)

@task
def docker_build(ctx: Context, progress: str = "plain") -> None:
    """Build docker images."""
    docker_build_train(ctx, progress)
    docker_build_api(ctx, progress)
    docker_build_frontend(ctx, progress)

@task
def docker_build_train(ctx: Context, progress: str = "plain") -> None:
    """Build Train docker image."""
    ctx.run(
        f"docker build -t train:latest . -f dockerfiles/train.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_build_api(ctx: Context, progress: str = "plain") -> None:
    """Build API docker image."""
    ctx.run(
        f"docker build -t api:latest . -f dockerfiles/api.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_build_frontend(ctx: Context, progress: str = "plain") -> None:
    """Build Frontend docker image."""
    ctx.run(
        f"docker build -t frontend:latest . -f dockerfiles/frontend.dockerfile --progress={progress}",
        echo=True,
        pty=not WINDOWS
    )

@task
def docker_run_frontend(ctx: Context) -> None:
    """Run Frontend docker container."""
    ctx.run("docker run -p 8080:8080 -e PORT=8080 frontend:latest", echo=True, pty=not WINDOWS)

@task
def docker_run_api(ctx: Context) -> None:
    """Run API docker container."""
    ctx.run("docker run -p 8000:8000 api:latest", echo=True, pty=not WINDOWS)

# Documentation commands
@task(dev_requirements)
def build_docs(ctx: Context) -> None:
    """Build documentation."""
    ctx.run("mkdocs build --config-file docs/mkdocs.yaml --site-dir build", echo=True, pty=not WINDOWS)

@task(dev_requirements)
def serve_docs(ctx: Context) -> None:
    """Serve documentation."""
    ctx.run("mkdocs serve --config-file docs/mkdocs.yaml", echo=True, pty=not WINDOWS)
