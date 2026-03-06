from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import SixDSTAlgorithm


class MyAlgorithm(SixDSTAlgorithm):
    def __init__(self):
        super().__init__(pop_size=12, crossover_rate=0.7, mutate_rate=0.1)


data = DataLoader.load("Circuit")
monitor = Monitor()
algorithm = MyAlgorithm()
workflow = Workflow(algorithm, data, monitor=monitor, mode="s", verbose=False)
workflow.run(steps=10)
print(monitor.result())
