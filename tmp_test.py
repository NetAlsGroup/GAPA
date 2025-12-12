from gapa4 import StrategyPlan
import json

plan = StrategyPlan()
print(json.dumps(plan.to_dict(), indent=2))


from autoadapt.api.planner import StrategyPlan
import json

plan = StrategyPlan()
print(json.dumps(plan.to_dict(), indent=2))
