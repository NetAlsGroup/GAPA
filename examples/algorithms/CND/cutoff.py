from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import CutOffAlgorithm

data = DataLoader.load("Circuit")
monitor = Monitor()
algorithm = CutOffAlgorithm(pop_size=16)
workflow = Workflow(algorithm, data, monitor=monitor, mode="s", verbose=False)
workflow.run(steps=20)
print(monitor.result())
