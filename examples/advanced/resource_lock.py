from gapa import ResourceManager

manager = ResourceManager(timeout_s=15.0)
servers = manager.server()
remote_servers = [item for item in servers if item.get("id") != "local" and item.get("online")] if isinstance(servers, list) else []

if not remote_servers:
    print({"error": "no online remote servers available for resource lock"})
else:
    scope = remote_servers[0]["id"]
    lock_info = manager.lock_resource(scope=scope, duration_s=120, warmup_iters=1, mem_mb=1024)
    print(lock_info)
    print(manager.lock_status(scope=scope))

    results = lock_info.get("results", {}) if isinstance(lock_info, dict) else {}
    node = results.get(scope, {}) if isinstance(results, dict) else {}
    lock_id = node.get("lock_id") if isinstance(node, dict) else None
    owner = node.get("owner") if isinstance(node, dict) else None

    print(manager.renew_resource(scope=scope, duration_s=120, lock_id=lock_id, owner=owner))
    print(manager.release_resource(scope=scope))
