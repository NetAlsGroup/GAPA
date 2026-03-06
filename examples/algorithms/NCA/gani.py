from gapa import DataLoader, Monitor, Workflow
from gapa.algorithms import GANIAlgorithm

data = DataLoader.load("chameleon_filtered")
monitor = Monitor()
algorithm = GANIAlgorithm(pop_size=8)
workflow = Workflow(algorithm, data, monitor=monitor, mode="s", verbose=False)
workflow.run(steps=20)
print(monitor.result())
