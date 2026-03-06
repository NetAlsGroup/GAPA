# 运维与 QA

这个文档用于承接性能、稳定性、发布验证等内容，不再放在 README 的新手主路径里。

## 性能基线与回归门禁

- 生成性能基线（synthetic）：
  - `python examples/run_perf_baseline.py --profile small --source synthetic`
- 基于最小真实运行链路生成 live 基线：
  - `python examples/run_perf_baseline.py --profile small --source live --live-samples 48`
- 生成发布级基线（真实 workload 采样）：
  - `python examples/run_perf_baseline.py --profile release_small --source real --real-dataset ForestFire_n500 --real-generations 1 --real-pop-size 12 --real-runs 2`
- 重新采集当前指标并执行门禁对比：
  - `python tests/perf_regression_gate.py --baseline <baseline.json> --current <current.json> --output <gate.json>`

门禁覆盖 `S` / `SM` / `M` / `MNM` 的 mode 集合一致性、吞吐下降、时延上升、恢复时延上升和远程失败率漂移。

## 长稳与混沌稳定性

- 执行确定性 soak + chaos 验证：
  - `python tests/soak_chaos_stability.py --iterations 80 --output .multi-agents/qa/qa-soak-and-chaos-stability-hardening-iteration-14.json`
- 发布前继续执行跨平台 P0 门禁：
  - `python .multi-agents/scripts/run_cross_platform_mode_gate.py`

## MNM 通信验证

- 生成通信前后对比报告：
  - `python tests/mnm_comm_iteration16_report.py --output .multi-agents/qa/mnm-communication-optimization-iteration-16.json`
- QA 模板结果：
  - `.multi-agents/qa/qa-mnm-communication-algorithm-optimization-iteration-16.json`

## 发布候选包

- RC 清单：`docs/RC_CHECKLIST_ITERATION_15.md`
- 发布说明：`docs/RELEASE_NOTES_RC_ITERATION_15.md`
- 回滚手册：`docs/ROLLBACK_RUNBOOK_ITERATION_15.md`
- 推广候选：`docs/PROMOTION_CANDIDATES_ITERATION_15.md`
