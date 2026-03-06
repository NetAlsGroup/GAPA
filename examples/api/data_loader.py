from gapa import DataLoader

print(DataLoader.list(task="CND"))
print(DataLoader.describe("Circuit"))

data = DataLoader.load("Circuit")
print({"dataset": data.name, "nodes": data.nodes_num, "budget": data.k})
