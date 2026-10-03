"""Locust load test for the Multi-Agent Orchestrator.

Run (server must be up, e.g. `make docker-up` or local uvicorn):
    LOADTEST_API_KEY=mao_xxx locust -f tests/load/locustfile.py \
        --host http://localhost:8000 --headless -u 10 -r 2 -t 30s --csv docs/performance/run

Scenarios: chat turns (dominant), conversation listing, health checks.
"""

import os
import random

from locust import HttpUser, between, task

_MESSAGES = [
    "Hello there!",
    "What is 21 * 2?",
    "Tell me a joke",
    "What is the latest news about AI?",
    "What does our return policy say?",
    "How are you doing today?",
]


class ChatUser(HttpUser):
    wait_time = between(0.3, 1.2)

    def on_start(self) -> None:
        api_key = os.environ.get("LOADTEST_API_KEY", "")
        with self.client.post(
            "/v1/auth/token", json={"api_key": api_key}, catch_response=True
        ) as response:
            if response.status_code == 200:
                self.token = response.json()["access_token"]
                self.headers = {"Authorization": f"Bearer {self.token}"}
            else:
                response.failure(f"token exchange failed: {response.status_code}")
                self.headers = {}

    @task(8)
    def chat(self):
        message = random.choice(_MESSAGES)
        with self.client.post(
            "/v1/chat", json={"message": message}, headers=self.headers, catch_response=True
        ) as response:
            if response.status_code == 200:
                body = response.json()
                if not body.get("response"):
                    response.failure("empty response")
            elif response.status_code == 429:
                response.failure("rate limited (429)")
            elif response.status_code == 409:
                response.failure("conversation conflict (409)")

    @task(1)
    def list_conversations(self):
        self.client.get("/v1/conversations", headers=self.headers)

    @task(1)
    def health(self):
        self.client.get("/health")
