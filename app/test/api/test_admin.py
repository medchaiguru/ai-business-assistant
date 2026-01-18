import os
from collections.abc import Generator
from unittest.mock import patch

import chromadb
import pytest
import requests
from chromadb.api import ClientAPI
from chromadb.errors import ChromaError
from fastapi.testclient import TestClient
from testcontainers.core.container import DockerContainer
from testcontainers.core.network import Network
from testcontainers.core.waiting_utils import wait_container_is_ready

from app.main import app

client = TestClient(app)

def test_trigger_indexing() -> None:
    """Test that the indexing endpoint triggers a background task."""
    # We patch the 'add_task' method of BackgroundTasks to verify it was called
    # But since TestClient runs synchronously, we can just check the response
    # and mock the actual task function to prevent it from running
    # if it were synchronous.

    with patch("app.api.routers.admin.run_indexing_task") as mock_task:
        response = client.post("/admin/index")

        assert response.status_code == 200
        assert response.json() == {
            "status": "accepted",
            "message": "Indexing started in background"
        }

        mock_task.assert_called_once()
        # Or check what it was called with:
        mock_task.assert_called_once_with(app)

        # Note: TestClient usually executes background tasks immediately.
        # So we can check if our mock was called.
        # However, since we pass the function reference to add_task,
        # verifying it was called requires deeper mocking of BackgroundTasks
        # or just trusting the response code for this unit test.

def load_env_file(filepath: str) -> dict[str, str]:
    """Parse .env file into a dictionary."""
    env_vars = {}
    if os.path.exists(filepath):
        with open(filepath, encoding="utf-8") as f:
            for line in f:
                # Ignore comments and empty lines
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                # Split key-value
                if "=" in line:
                    key, value = line.split("=", 1)
                    env_vars[key.strip()] = value.strip()
    return env_vars

@wait_container_is_ready(ChromaError, ValueError)
def _get_chroma_client(host: str, port: str) -> ClientAPI:
    """Wait for Chroma container to be ready by checking heartbeat."""
    client = chromadb.HttpClient(host=host, port=int(port))
    client.heartbeat()
    return client

@wait_container_is_ready(requests.RequestException, ValueError)
def _check_app_status(container: DockerContainer) -> None:
    """Simple test to check if the app is running."""
    host = container.get_container_host_ip()
    port = container.get_exposed_port(8000)
    response = requests.get(f"http://{host}:{port}/health", timeout=2)
    assert response.status_code == 200


@pytest.fixture(scope="module")
def env_file() -> dict[str, str]:
    """Path to .env file for Docker container."""
    env_file = ".env"
    return load_env_file(env_file)

@pytest.fixture(scope="module")
def network() -> Generator[Network, None, None]:
    """Create a shared network for containers to communicate."""
    with Network() as net:
        yield net

@pytest.fixture(scope="module")
def chroma_container(network: Network) -> Generator[DockerContainer, None, None]:
    """Spin up a ChromaDB container for testing."""
    container = DockerContainer("ghcr.io/chroma-core/chroma:latest")
    container.with_exposed_ports(8000)
    container.with_network(network)
    container.with_network_aliases("chroma-db")
    with container as chroma:
        # client = _get_chroma_client(
        #     host=chroma.get_container_host_ip(),
        #     port=chroma.get_exposed_port(8000)
        # )
        yield chroma

@pytest.fixture(scope="module")
def app_container(chroma_container: DockerContainer, env_file: dict[str, str]) -> Generator[DockerContainer, None, None]:
    """Spin up the FastAPI app in a Docker container."""
    container = DockerContainer("aisupport:latest")  # Use your built image
    container.with_exposed_ports(8000)
    container.with_network(chroma_container._network)
    container.with_network_aliases("this_app")
    if env_file:
        for api_key, value in env_file.items():
            container.with_env(api_key, value)

    # OVERRIDE Chroma settings to point to the other container by alias
    container.with_env("CHROMA_SERVER_HOST", "chroma-db")
    container.with_env("CHROMA_SERVER_HTTP_PORT", "8000")

    with container as this_app:
        # Wait for app to be ready
        _check_app_status(this_app)
        yield this_app
        # This runs after all tests in this module finish (or if startup fails)
        print("\n=== CONTAINER LOGS ===\n")
        logs = this_app.get_logs()
        if isinstance(logs, bytes):
            print(logs.decode("utf-8", errors="replace"))
        else:
            # logs might be a tuple (stdout, stderr) depending on version
            print(logs[0].decode("utf-8", errors="replace"))
            print(logs[1].decode("utf-8", errors="replace"))
        print("\n======================\n")

@pytest.fixture
def app_url(app_container: DockerContainer) -> str:
    """Get the base URL of the running app."""
    host = app_container.get_container_host_ip()
    port = app_container.get_exposed_port(8000)
    return f"http://{host}:{port}"


def test_health_check(app_url: str) -> None:
    """Test health endpoint on running Docker container."""
    response = requests.get(f"{app_url}/health", timeout=2)
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_index_data(app_url: str) -> None:
    """Helper to trigger indexing via admin endpoint."""
    response = requests.post(f"{app_url}/admin/index", timeout=5)
    assert response.status_code == 200
    assert response.json()["status"] == "accepted"
