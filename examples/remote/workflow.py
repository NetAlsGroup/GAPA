from gapa import DataLoader, Monitor, ResourceManager, Workflow
from gapa.algorithms import SixDSTAlgorithm


def main() -> None:
    manager = ResourceManager()
    servers = manager.server()
    remote_servers = [item for item in servers if item.get("id") != "local"] if isinstance(servers, list) else []

    if not remote_servers:
        print({"error": "no remote server configured in servers.json"})
        return

    data = DataLoader.load("Circuit")
    monitor = Monitor()
    algorithm = SixDSTAlgorithm(pop_size=16)

    workflow = Workflow(
        algorithm=algorithm,
        data_loader=data,
        monitor=monitor,
        mode="m",
        remote_server=remote_servers[0]["id"],
        remote_devices=[0, 1, 2, 3],
        remote_use_strategy_plan=False,
        verbose=True,
    )
    workflow.run(steps=20)
    print(monitor.result())


if __name__ == "__main__":
    main()
