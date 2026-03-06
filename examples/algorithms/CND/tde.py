from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import TDEAlgorithm

data = DataLoader.load("Circuit")
monitor = Monitor()
algorithm = TDEAlgorithm(pop_size=16)
workflow = Workflow(algorithm, data, monitor=monitor, mode="s", verbose=False)
workflow.run(steps=20)
print(monitor.result())
