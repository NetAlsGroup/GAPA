from gapa import ResourceManager

manager = ResourceManager()
print(manager.resources(all_servers=True))
print(manager.strategy_plan(algorithm="SixDST", dataset="Circuit", mode="s", warmup=1))
