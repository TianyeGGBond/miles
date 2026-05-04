import argparse
import asyncio
import json
import logging

import httpx
import setproctitle
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.responses import Response

from miles.utils.misc import load_function

logger = logging.getLogger(__name__)


def run_router(args):
    """
    Run the Miles router with the specified configuration.
    """
    # Visible to `pkill -9 miles`; without this the daemon inherits "python".
    setproctitle.setproctitle("miles-router")

    # Initialize the router with tokenizer and lazy worker initialization
    miles_router = MilesRouter(args, verbose=False)

    # Start the server
    uvicorn.run(miles_router.app, host=args.sglang_router_ip, port=args.sglang_router_port, log_level="info")


class MilesRouter:
    def __init__(self, args, verbose=False):
        """Initialize the miles-router with SGLang router address"""
        self.args = args
        self.verbose = verbose

        self.app = FastAPI()
        self.app.router.on_startup.append(self._start_background_health_check)

        # F3 admission lifecycle (scope F39 / F14):
        #   - worker_request_counts: URL → in-flight count (also doubles as the
        #     "registered" set; presence in this dict ⇔ the worker has been
        #     declared to the router via add_worker, regardless of admission).
        #   - worker_failure_counts: URL → consecutive health-check failures.
        #   - dead_workers: quarantined URLs (health-check threshold exceeded).
        #   - enabled_workers: URLs admitted for routing. Source of truth for
        #     dispatch, NOT metadata. shrink/disable removes; expand/enable
        #     adds. _use_url selects from `enabled_workers - dead_workers`.
        #   - worker_engine_index_map: URL → engine_index, populated at
        #     add_worker and consumed by F3 metadata injection (iter 8).
        # URL -> Active Request Count (load state)
        self.worker_request_counts: dict[str, int] = {}
        # URL -> Consecutive Failures
        self.worker_failure_counts: dict[str, int] = {}
        # Quarantined workers excluded from routing pool
        self.dead_workers: set[str] = set()
        # Admitted-for-routing set. Standalone path adds every worker to this
        # set on add_worker (preserves legacy behavior); RLix-mode flow uses
        # disable/enable to flip admission without removing from the registry.
        self.enabled_workers: set[str] = set()
        self.worker_engine_index_map: dict[str, int] = {}
        self.max_weight_version = None

        max_connections = getattr(args, "miles_router_max_connections", None)
        if max_connections is None:
            max_connections = (
                args.sglang_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
            )

        timeout = getattr(args, "miles_router_timeout", None)

        self.client = httpx.AsyncClient(
            limits=httpx.Limits(max_connections=max_connections),
            timeout=httpx.Timeout(timeout),
        )

        self._setup_routes()

        for middleware_path in args.miles_router_middleware_paths or []:
            if self.verbose:
                print(f"[miles-router] Loading middleware from: {middleware_path}")
            middleware = load_function(middleware_path)
            self.app.add_middleware(middleware, router=self)

    def _setup_routes(self):
        """Setup all the HTTP routes except catch-all proxy"""
        # sglang-router api
        self.app.post("/add_worker")(self.add_worker)
        self.app.get("/list_workers")(self.list_workers)
        # F3 admission lifecycle endpoints (RLix-mode shrink/expand).
        self.app.post("/disable_worker")(self.disable_worker)
        self.app.post("/enable_worker")(self.enable_worker)
        self.app.post("/remove_worker")(self.remove_worker)
        # Catch-all route for proxying to SGLang - must be registered LAST
        self.app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])(self.proxy)

    async def _start_background_health_check(self):
        asyncio.create_task(self._health_check_loop())

    async def _check_worker_health(self, url):
        """Encapsulated health check logic for better maintainability."""
        try:
            response = await self.client.get(f"{url}/health", timeout=5.0)
            if response.status_code == 200:
                return url, True
            logger.debug(f"[miles-router] Worker {url} is unhealthy (Status: {response.status_code})")
        except Exception as e:
            logger.debug(f"[miles-router] Worker {url} health check failed: {e}")
        return url, False

    async def _health_check_loop(self):
        """Background loop to monitor worker health and adjust routing pool."""
        interval = self.args.rollout_health_check_interval
        threshold = self.args.miles_router_health_check_failure_threshold

        while True:
            try:
                await asyncio.sleep(interval)

                # Probe only enabled, non-dead workers. Disabled workers are
                # parked (no traffic should reach them); failing health on
                # them would fight the F2 disable lifecycle. Health-driven
                # quarantine still applies once an enabled worker accumulates
                # enough consecutive failures.
                urls = [u for u in self.enabled_workers if u not in self.dead_workers]
                if not urls:
                    continue

                results = await asyncio.gather(*(self._check_worker_health(url) for url in urls))

                for url, is_healthy in results:
                    if not is_healthy:
                        failures = self.worker_failure_counts.get(url, 0) + 1
                        self.worker_failure_counts[url] = failures

                        if failures >= threshold:
                            logger.warning(
                                f"[miles-router] Worker {url} failed {threshold} consecutive health checks. Marking as DEAD."
                            )
                            self.dead_workers.add(url)
                            # TODO (chenyang): Connect back 'dead' workers requires a mechanism to sync
                            # model versions to avoid off-policy issues from stale weights, since these
                            # dead workers' parameters may not be refitted.
                    else:
                        self.worker_failure_counts[url] = 0

                logger.debug(
                    f"[miles-router] Health check complete. {len(self.worker_request_counts) - len(self.dead_workers)} workers healthy."
                )

            except asyncio.CancelledError:
                logger.warning("[miles-router] Background health check loop is being cancelled.")
                raise
            except Exception as e:
                logger.error(f"[miles-router] Unexpected error in health check loop: {e}", exc_info=True)
                await asyncio.sleep(5)

    async def proxy(self, request: Request, path: str):
        """Proxy all other requests to the SGLang router"""
        result = await self.do_proxy(request, path)
        return self.build_proxy_response(result)

    async def do_proxy(
        self,
        request: Request,
        path: str,
        body: bytes | None = None,
        headers: dict | None = None,
    ) -> dict:
        """Core proxy logic. Returns dict with request_body, response_body, status_code, headers."""
        worker_url = self._use_url()
        url = f"{worker_url}/{path}"

        if body is None:
            body = await request.body()
        if headers is None:
            headers = dict(request.headers)
        if body is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}

        try:
            response = await self.client.request(request.method, url, content=body, headers=headers)
            content = await response.aread()
            return {
                "request_body": body,
                "response_body": content,
                "status_code": response.status_code,
                "headers": dict(response.headers),
            }
        finally:
            self._finish_url(worker_url)

    def build_proxy_response(self, result: dict) -> Response:
        """Build HTTP response from proxy result."""
        content = result["response_body"]
        status_code = result["status_code"]
        headers = result["headers"]
        headers = {k: v for k, v in headers.items() if k.lower() not in ("content-length", "transfer-encoding")}
        content_type = headers.get("content-type", "")
        try:
            data = json.loads(content)
            return JSONResponse(content=data, status_code=status_code, headers=headers)
        except Exception:
            return Response(content=content, status_code=status_code, headers=headers, media_type=content_type)

    async def add_worker(self, request: Request):
        """Add a new worker to the router.
        Supports providing the URL via query string or JSON body.
        Examples:
        - POST /add_worker?url=http://127.0.0.1:10090
        - POST /add_worker?url=http://127.0.0.1:10090&engine_index=0
        - POST /add_worker  with body {"url": "...", "engine_index": 0}
        """
        worker_url, engine_index = self._extract_worker_params(
            request.query_params, await self._safe_json_body(request)
        )
        if not worker_url:
            return JSONResponse(
                status_code=400, content={"error": "worker_url is required (use query ?url=... or JSON body)"}
            )
        self._add_worker_internal(worker_url, engine_index)
        return {"status": "success", "worker_urls": self.worker_request_counts}

    async def disable_worker(self, request: Request):
        """Close admission for a worker without removing it from the registry.

        Used by F2 RolloutManager.shrink_engines: the engine handle and
        worker_request_counts entry persist (so balance accounting stays
        consistent with later _finish_url calls); only enabled_workers
        loses the URL.
        """
        worker_url, _ = self._extract_worker_params(
            request.query_params, await self._safe_json_body(request)
        )
        if not worker_url:
            return JSONResponse(status_code=400, content={"error": "worker_url is required"})
        self._disable_worker_internal(worker_url)
        return {"status": "success", "enabled_workers": sorted(self.enabled_workers)}

    async def enable_worker(self, request: Request):
        """Re-open admission for a previously-disabled worker.

        Reset failure_count to 0 (per F68 — invariant: re-admit must not
        carry over a stale failure count from a prior disable cycle).
        """
        worker_url, _ = self._extract_worker_params(
            request.query_params, await self._safe_json_body(request)
        )
        if not worker_url:
            return JSONResponse(status_code=400, content={"error": "worker_url is required"})
        self._enable_worker_internal(worker_url)
        return {"status": "success", "enabled_workers": sorted(self.enabled_workers)}

    async def remove_worker(self, request: Request):
        """Permanently drop a worker from every registry.

        Distinct from disable_worker: removes from worker_request_counts /
        worker_failure_counts / enabled_workers / dead_workers /
        worker_engine_index_map. Use only on actor death; routing-time
        shrink uses disable_worker to preserve in-flight balance.
        """
        worker_url, _ = self._extract_worker_params(
            request.query_params, await self._safe_json_body(request)
        )
        if not worker_url:
            return JSONResponse(status_code=400, content={"error": "worker_url is required"})
        self._remove_worker_internal(worker_url)
        return {"status": "success", "worker_urls": sorted(self.worker_request_counts)}

    async def list_workers(self, request: Request):
        """List all registered workers"""
        return {"urls": list(self.worker_request_counts.keys())}

    # ------------------------------------------------------------------
    # F3 admission lifecycle helpers — stay sync per scope F14. Only the
    # async endpoints (and iter 7's _health_check_loop notify edge) carry
    # any awaitable work.
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_worker_params(
        query_params, body_payload: dict | None
    ) -> tuple[str | None, int | None]:
        """Pull `worker_url` (and optional `engine_index`) from query or body."""
        worker_url = query_params.get("url") or query_params.get("worker_url")
        engine_index_raw = query_params.get("engine_index")
        if not worker_url and body_payload:
            worker_url = body_payload.get("url") or body_payload.get("worker_url")
        if engine_index_raw is None and body_payload:
            engine_index_raw = body_payload.get("engine_index")
        engine_index: int | None = None
        if engine_index_raw is not None:
            engine_index = int(engine_index_raw)
        return worker_url, engine_index

    @staticmethod
    async def _safe_json_body(request: Request) -> dict | None:
        body = await request.body()
        if not body:
            return None
        try:
            return json.loads(body)
        except (ValueError, TypeError):
            return None

    def _add_worker_internal(self, url: str, engine_index: int | None) -> None:
        """Register a worker, default-admit it, clear stale dead-state.

        F68 invariants: ``setdefault`` so re-add doesn't zero an in-flight
        count; ``dead_workers.discard`` so re-registering the same URL
        doesn't carry old `dead` poisoning into the new lifecycle.
        """
        self.worker_request_counts.setdefault(url, 0)
        self.worker_failure_counts.setdefault(url, 0)
        self.dead_workers.discard(url)
        self.enabled_workers.add(url)
        if engine_index is not None:
            self.worker_engine_index_map[url] = engine_index
        if self.verbose:
            print(f"[miles-router] Added worker: {url} (engine_index={engine_index})")

    def _remove_worker_internal(self, url: str) -> None:
        """Drop the URL from every per-worker registry."""
        self.worker_request_counts.pop(url, None)
        self.worker_failure_counts.pop(url, None)
        self.dead_workers.discard(url)
        self.enabled_workers.discard(url)
        self.worker_engine_index_map.pop(url, None)
        if self.verbose:
            print(f"[miles-router] Removed worker: {url}")

    def _disable_worker_internal(self, url: str) -> None:
        """Close admission. F68: reset failure_count so a sleep cycle does
        not poison the next enable.
        """
        self.enabled_workers.discard(url)
        self.worker_failure_counts[url] = 0
        if self.verbose:
            print(f"[miles-router] Disabled worker: {url}")

    def _enable_worker_internal(self, url: str) -> None:
        """Re-open admission. F68: reset failure_count + dead_workers entry."""
        if url not in self.worker_request_counts:
            # Defensive: enable on an unknown URL is a no-op rather than
            # an error — caller may race with disable/remove.
            return
        self.worker_failure_counts[url] = 0
        self.dead_workers.discard(url)
        self.enabled_workers.add(url)
        if self.verbose:
            print(f"[miles-router] Enabled worker: {url}")

    def _use_url(self):
        """Select an admitted worker URL with minimal active requests.

        Source of truth for dispatch is ``enabled_workers - dead_workers``
        (per F39 critical invariant — NOT just worker_request_counts
        membership). Iter 7 wraps this in an asyncio.Condition for the
        C20 0-active suspend; iter 6 keeps the pre-C20 sync raise so
        existing standalone behavior is unchanged.
        """
        candidates = self.enabled_workers - self.dead_workers
        if not candidates:
            raise RuntimeError("No enabled live workers available in the pool")
        url = min(candidates, key=lambda u: self.worker_request_counts.get(u, 0))
        self.worker_request_counts[url] += 1
        return url

    def _finish_url(self, url):
        """Mark the request to the given URL as finished"""
        if url not in self.worker_request_counts:
            # remove_worker may have raced; tolerate.
            return
        self.worker_request_counts[url] -= 1
        if self.worker_request_counts[url] < 0:
            # The negative check is tight in standalone but in RLix mode the
            # disable→remove→re-add ordering could theoretically race. Keep
            # the assert form so the test suite catches regressions.
            raise AssertionError(f"URL {url} count went negative")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--sglang-host", type=str, required=True)
    parser.add_argument("--sglang-port", type=int, required=True)
    parser.add_argument("--tokenizer-name", type=str, help="Name of the tokenizer to use for tokenization")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose output")

    args = parser.parse_args()

    # Run the router
    run_router(args)
