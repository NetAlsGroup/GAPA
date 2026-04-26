from gapa import ResourceManager


def main() -> None:
    manager = ResourceManager()
    print(manager.resources(all_servers=True))
    print(manager.strategy_plan(algorithm="SixDST", dataset="Circuit", mode="s", warmup=1))


if __name__ == "__main__":
    main()
