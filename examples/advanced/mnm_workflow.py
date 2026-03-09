from gapa import DataLoader, Monitor, ResourceManager, Workflow
from gapa.algorithms import SixDSTAlgorithm

manager = ResourceManager()
servers = manager.server()
remote_servers = [item["id"] for item in servers if item.get("id") != "local" and item.get("online")] if isinstance(servers, list) else []

if not remote_servers:
    print({"error": "no online remote servers configured in .env / GAPA_REMOTE_SERVERS"})
else:
    data = DataLoader.load("Circuit")
    monitor = Monitor()
    workflow = Workflow(
        algorithm=SixDSTAlgorithm(),
        data_loader=data,
        monitor=monitor,
        mode="mnm",
        servers=remote_servers,
        verbose=True,
    )
    workflow.run(steps=200)
    print(monitor.result())
