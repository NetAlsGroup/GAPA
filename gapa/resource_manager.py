from __future__ import annotations

from typing import Any, Dict, List, Optional

from gapa.workflow import Monitor


class ResourceManager:
    """Public resource access surface separated from Monitor state tracking."""

    def __init__(self, api_base: Optional[str] = None, timeout_s: float = 5.0):
        self.api_base = api_base
        self.timeout_s = float(timeout_s)
        self._monitor = Monitor(api_base=api_base, timeout_s=timeout_s)

    def server(self) -> Any:
        return self._monitor.server()

    def server_resource(self, server_id_or_name: str) -> Dict[str, Any]:
        return self._monitor.server_resource(server_id_or_name)

    def resources(self, all_servers: bool = True) -> Dict[str, Any]:
        return self._monitor.resources(all_servers=all_servers)

    def lock_status(self, scope: str = "all", realtime: bool = True) -> Dict[str, Any]:
        return self._monitor.lock_status(scope=scope, realtime=realtime)

    def lock_resource(
        self,
        scope: str = "all",
        duration_s: float = 600.0,
        warmup_iters: int = 2,
        mem_mb: int = 1024,
        strict_idle: bool = False,
        devices: Optional[List[Any]] = None,
        devices_by_server: Optional[Dict[str, List[Any]]] = None,
    ) -> Dict[str, Any]:
        return self._monitor.lock_resource(
            scope=scope,
            duration_s=duration_s,
            warmup_iters=warmup_iters,
            mem_mb=mem_mb,
            strict_idle=strict_idle,
            devices=devices,
            devices_by_server=devices_by_server,
        )

    def renew_resource(
        self,
        scope: str = "all",
        duration_s: Optional[float] = None,
        lock_id: Optional[str] = None,
        owner: Optional[str] = None,
    ) -> Dict[str, Any]:
        return self._monitor.renew_resource(
            scope=scope,
            duration_s=duration_s,
            lock_id=lock_id,
            owner=owner,
        )

    def release_resource(self, scope: str = "all") -> Dict[str, Any]:
        return self._monitor.unlock_resource(scope=scope)

    def strategy_plan(
        self,
        server_id: Optional[str] = None,
        algorithm: str = "SixDST",
        dataset: str = "A01",
        mode: str = "s",
        warmup: int = 1,
    ) -> Dict[str, Any]:
        return self._monitor.strategy_plan(
            server_id=server_id,
            algorithm=algorithm,
            dataset=dataset,
            mode=mode,
            warmup=warmup,
        )

    def distributed_strategy_plan(
        self,
        server_ids: List[str],
        algorithm: str = "SixDST",
        dataset: str = "A01",
        mode: str = "mnm",
        warmup: int = 1,
    ) -> Dict[str, Any]:
        return self._monitor.distributed_strategy_plan(
            server_ids=server_ids,
            algorithm=algorithm,
            dataset=dataset,
            mode=mode,
            warmup=warmup,
        )

    def resource_rows(self, resources: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
        return self._monitor.resource_rows(resources=resources)

    def resource_dataframe(self, rows: Optional[List[Dict[str, Any]]] = None):
        return self._monitor.resource_dataframe(rows=rows)

    def plot_resources(self, metric: str = "cpu_usage_percent", rows: Optional[List[Dict[str, Any]]] = None):
        return self._monitor.plot_resources(metric=metric, rows=rows)
