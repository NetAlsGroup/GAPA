# 运维与 QA

这个文档用于承接性能、稳定性、发布验证等内容，不再放在 README 的新手主路径里。

文档导航：

- [README.zh-CN.md](README.zh-CN.md)

## 性能基线与回归门禁

- 公开工作流中的最小验证路径保持为：
  - `python examples/api/workflow.py`
  - `python examples/algorithms/CND/sixdst.py`
- 如果你维护自己的性能基线，请在本地验证流程中与私有发布阈值进行对比。

回归阈值、内部门禁和治理自动化留在内部交付流程中。

## 长稳与混沌稳定性

- 公开验证路径建议使用当前维护的 API 与算法样例：
  - `python examples/api/workflow.py`
  - `python examples/algorithms/CND/sixdst.py`

长稳、chaos 和发布级门禁属于内部交付流程，不再要求开源用户执行。

## MNM 通信验证

公开工作流中的 MNM 验证建议从资源视图和分布式启动入口开始：

- `python examples/api/resource_manager.py`
- `python server_agent.py`

发布级 MNM 通信压测和 QA 打包保留在内部流程中。

## 发布候选包

- RC 清单：`docs/RC_CHECKLIST_ITERATION_15.md`
- 发布说明：`docs/RELEASE_NOTES_RC_ITERATION_15.md`
- 回滚手册：`docs/ROLLBACK_RUNBOOK_ITERATION_15.md`
- 推广候选：`docs/PROMOTION_CANDIDATES_ITERATION_15.md`

## Onboarding 一致性门禁

当前维护的公开用户主路径由以下内容组成：

- `README.zh-CN.md`
- `examples/api/`
- `examples/algorithms/`

内部 onboarding 一致性门禁不再作为公开仓库工作流的一部分。
