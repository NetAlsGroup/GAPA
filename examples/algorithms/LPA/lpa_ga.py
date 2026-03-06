from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import LPAGAAlgorithm

data = DataLoader.load("dolphins")
monitor = Monitor()
algorithm = LPAGAAlgorithm(pop_size=16)
workflow = Workflow(algorithm, data, monitor=monitor, mode="s", verbose=False)
workflow.run(steps=20)
print(monitor.result())
