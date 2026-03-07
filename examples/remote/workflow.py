from gapa import DataLoader, Monitor, ResourceManager, Workflow
from gapa.algorithms import SixDSTAlgorithm

manager = ResourceManager()
servers = manager.server()
remote_servers = [item for item in servers if item.get("id") != "local"] if isinstance(servers, list) else []

if not remote_servers:
    print({"error": "no remote server configured in .env / GAPA_REMOTE_SERVERS"})
else:
    data = DataLoader.load("Circuit")
    monitor = Monitor()
    algorithm = SixDSTAlgorithm()

    workflow = Workflow(
        algorithm=algorithm,
        data_loader=data,
        monitor=monitor,
        mode="m",
        remote_server=remote_servers[0]["id"],
        remote_use_strategy_plan=True,
        verbose=True,
    )
    workflow.run(steps=10)
    print(monitor.result())
