from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import SixDSTAlgorithm


def main() -> None:
    data = DataLoader.load("Circuit")
    monitor = Monitor()
    algorithm = SixDSTAlgorithm(pop_size=16)
    workflow = Workflow(algorithm, data, monitor=monitor, mode="m", verbose=False)
    workflow.run(steps=20)
    print(monitor.result())


if __name__ == "__main__":
    main()
