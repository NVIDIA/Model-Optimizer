"""Lazy aiohttp transport helpers.

``aiohttp`` is intentionally an example-local dependency so importing the
distributed-evaluation schemas does not require it.
"""

from __future__ import annotations

from typing import Any


class HttpStatusError(RuntimeError):
    def __init__(self, status: int, message: str, payload: dict[str, Any] | None = None):
        super().__init__(f"HTTP {status}: {message}")
        self.status = status
        self.payload = payload or {}


class AsyncHttpClient:
    def __init__(
        self,
        *,
        token: str,
        connect_timeout_seconds: float = 10.0,
        task_timeout_seconds: float = 7200.0,
    ):
        self.token = token
        self.connect_timeout_seconds = connect_timeout_seconds
        self.task_timeout_seconds = task_timeout_seconds
        self._session = None

    async def _get_session(self):
        if self._session is None or self._session.closed:
            try:
                import aiohttp
            except ImportError as error:
                raise RuntimeError(
                    "Distributed evaluation requires aiohttp. Install "
                    "examples/puzzletron/requirements.txt."
                ) from error
            timeout = aiohttp.ClientTimeout(
                total=self.task_timeout_seconds,
                connect=self.connect_timeout_seconds,
                sock_connect=self.connect_timeout_seconds,
                sock_read=self.task_timeout_seconds,
            )
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def request(
        self,
        method: str,
        url: str,
        *,
        payload: dict[str, Any] | None = None,
        authenticated: bool = True,
    ) -> dict[str, Any]:
        session = await self._get_session()
        headers = {}
        if authenticated:
            headers["Authorization"] = f"Bearer {self.token}"
        async with session.request(method, url, json=payload, headers=headers) as response:
            try:
                data = await response.json()
            except Exception:
                data = {"message": await response.text()}
            if response.status >= 400:
                raise HttpStatusError(
                    response.status,
                    str(data.get("message") or data.get("error") or response.reason),
                    data,
                )
            return data

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()


def require_aiohttp():
    try:
        import aiohttp  # noqa: F401
    except ImportError as error:
        raise RuntimeError(
            "Distributed evaluation requires aiohttp. Install "
            "examples/puzzletron/requirements.txt."
        ) from error
