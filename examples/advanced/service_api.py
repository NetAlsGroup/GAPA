from gapa.config import get_app_base_url

try:
    import requests
except Exception:
    requests = None

base_url = get_app_base_url().rstrip("/")

if requests is None:
    print({"error": "requests not available"})
else:
    session = requests.Session()
    session.trust_env = False
    results = []

    def fetch(path: str):
        try:
            response = session.get(f"{base_url}{path}", timeout=10)
        except Exception as exc:
            return {"error": str(exc), "path": path}
        try:
            body = response.json()
        except Exception:
            body = {"raw": response.text}
        return {"status_code": response.status_code, "body": body, "path": path}

    print({"base_url": base_url})
    for path in (
        "/api/servers",
        "/api/v1/resources/all",
        "/api/datasets",
        "/api/analysis/status",
        "/api/analysis/queue",
        "/api/transport/metrics",
    ):
        item = fetch(path)
        results.append(item)
        print(item)

    statuses = [item.get("status_code") for item in results if isinstance(item, dict) and "status_code" in item]
    if statuses and all(code == 403 for code in statuses):
        print(
            {
                "diagnostic": "all service endpoints returned HTTP 403",
                "hint": (
                    "service_api.py is likely not talking to the GAPA app.py instance. "
                    "Check GAPA_APP_HOST/GAPA_APP_PORT in .env and confirm app.py is running on that exact address."
                ),
                "expected_base_url": base_url,
            }
        )
