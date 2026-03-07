from gapa import ResourceManager

manager = ResourceManager()
servers = manager.server()
remote_servers = [item for item in servers if item.get("id") != "local"] if isinstance(servers, list) else []

print(servers)

if remote_servers:
    remote_id = remote_servers[0]["id"]
    print(manager.server_resource(remote_id))
    print(manager.strategy_plan(server_id=remote_id, algorithm="SixDST", dataset="Circuit", mode="s", warmup=0, timeout_s=30))
else:
    print({"error": "no remote server configured in .env / GAPA_REMOTE_SERVERS"})
