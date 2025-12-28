const { createApp } = Vue;
    const app = createApp({
      data() {
        return {
          view: (() => {
            try {
              const saved = localStorage.getItem("gapa_view");
              return saved || "monitor";
            } catch (e) {
              return "monitor";
            }
          })(),
        };
      },
      methods: {
        setView(view) {
          this.view = view;
          try { localStorage.setItem("gapa_view", view); } catch (e) {}
        },
      },
      mounted() {
        if (!document.getElementById(`view-${this.view}`)) {
          this.view = "monitor";
        }
      },
    });
    const appVm = app.mount("#app");

    const cardsDiv = document.getElementById("cards");
    const serverSelect = document.getElementById("select-server");
    const serverSelectMulti = document.getElementById("select-server-multi");
    const serverSelectMultiMode = document.getElementById("select-server-multi-mode");
    const serverSelectMultiList = document.getElementById("select-server-multi-list");
    const serverSelectMultiContainer = document.getElementById("multi-options-container");
    const algoSelect = document.getElementById("select-algo");
    const evalDataset = document.getElementById("select-dataset");
    const warmupWrap = document.getElementById("warmup-wrap");
    const warmupInput = document.getElementById("input-warmup");
    const tpeTrialsInput = document.getElementById("input-tpe-trials");
    const tpeWarmupInput = document.getElementById("input-tpe-warmup");
    const gpuBusyInput = document.getElementById("input-gpu-busy");
    const gpuFreeInput = document.getElementById("input-gpu-free");
    const logBox = document.getElementById("log-box");
    const gaPanel = document.getElementById("ga-panel");

    const modal = document.getElementById("modal");
    const btnOpenModal = document.getElementById("btn-open-modal");
    const btnCloseModal = document.getElementById("btn-close-modal");
    const btnCancel = document.getElementById("btn-cancel");
    const btnAdd = document.getElementById("btn-add");
    const evalModal = document.getElementById("eval-modal");
    const btnOpenEval = document.getElementById("btn-open-eval");
    const btnEvalClose = document.getElementById("btn-eval-close");
    const evalNext1 = document.getElementById("eval-next-1");
    const evalNext2 = document.getElementById("eval-next-2");
    const evalPrev2 = document.getElementById("eval-prev-2");
    const evalPrev3 = document.getElementById("eval-prev-3");
    const evalSingleWrap = document.getElementById("eval-single-wrap");
    const evalMultiWrap = document.getElementById("eval-multi-wrap");
    const evalSummary = document.getElementById("eval-summary");
    const summaryMode = document.getElementById("summary-mode");
    const summaryServer = document.getElementById("summary-server");
    const summaryAlgo = document.getElementById("summary-algo");
    const summaryDataset = document.getElementById("summary-dataset");
    const summaryWarmup = document.getElementById("summary-warmup");
    const btnLock = document.getElementById("btn-lock");
    const lockModal = document.getElementById("lock-modal");
    const lockClose = document.getElementById("lock-close");
    const lockMode = document.getElementById("lock-mode");
    const lockScope = document.getElementById("lock-scope");
    const lockDuration = document.getElementById("lock-duration");
    const lockWarmup = document.getElementById("lock-warmup");
    const lockWarmupWrap = document.getElementById("lock-warmup-wrap");
    const lockMem = document.getElementById("lock-mem");
    const lockManualWrap = document.getElementById("lock-manual-wrap");
    const lockServerList = document.getElementById("lock-server-list");
    const lockApply = document.getElementById("lock-apply");
    const lockRelease = document.getElementById("lock-release");
    const lockStatusScope = document.getElementById("lock-status-scope");
    const lockStatusBox = document.getElementById("lock-status");
    const lockLogBox = document.getElementById("lock-log");
    const btnLockStatus = document.getElementById("btn-lock-status");
    const btnLockReleaseNow = document.getElementById("btn-lock-release-now");
    const btnLockLogClear = document.getElementById("btn-lock-log-clear");

    const sources = [{ id: "local", base: "", label: "本机", expanded: true, meta: { type: "local" } }];
    const snapshots = new Map();

    // Algo run DOM
    let RUN_SERVER_ID = "local";
    const runServer = document.getElementById("run-server");
    const runAlgo = document.getElementById("run-algo");
    const runDataset = document.getElementById("run-dataset");
    const runMode = document.getElementById("run-mode");
    const runGpuSingleWrap = document.getElementById("run-gpu-single-wrap");
    const runGpuSingle = document.getElementById("run-gpu-single");
    const runGpuMultiWrap = document.getElementById("run-gpu-multi-wrap");
    const runGpuMulti = document.getElementById("run-gpu-multi");
    const runIters = document.getElementById("run-iters");
    const runPc = document.getElementById("run-pc");
    const runPm = document.getElementById("run-pm");
    const btnRun = document.getElementById("btn-run");
    const btnRunStop = document.getElementById("btn-run-stop");
    const runState = document.getElementById("run-state");
    const runTaskId = document.getElementById("run-taskid");
    const runProgress = document.getElementById("run-progress");
    const runProgressText = document.getElementById("run-progress-text");
    const runBest = document.getElementById("run-best");
    const runComm = document.getElementById("run-comm");
    const runCommDetail = document.getElementById("run-comm-detail");
    const runLog = document.getElementById("run-log");
    const runChart = document.getElementById("run-chart");
    const btnRunClear = document.getElementById("btn-run-clear");
    const runLockStatus = document.getElementById("run-lock-status");
    const runLockDetail = document.getElementById("run-lock-detail");
    const btnRunLockRefresh = document.getElementById("btn-run-lock-refresh");
    const btnRunLockRelease = document.getElementById("btn-run-lock-release");
    const btnAnalyzeLock = document.getElementById("btn-analyze-lock");
    const RUN_CONFIG_KEY = "gapa_run_config_v1";

    // History DOM
    const historyList = document.getElementById("history-list");
    const historyDetail = document.getElementById("history-detail");
    const btnHistoryClear = document.getElementById("btn-history-clear");
    const btnHistoryDelete = document.getElementById("btn-history-delete");
    const historySelectAll = document.getElementById("history-select-all");
    const PLAN_KEY = "gapa_saved_plans_v1";
    let datasetsManifest = null;
    let lastEvalPlan = null;
    let lastEvalMode = null; // "single" | "distributed"
    let lastEvalTarget = null;
    let currentPoll = null;
    let lastLogCount = 0;

    function fmt(val, unit = "", fallback = "N/A") {
      return val == null ? fallback : `${val}${unit}`;
    }
    function pct(val) {
      return val == null ? "N/A" : `${Number(val).toFixed(1)}%`;
    }
    function log(line) {
      if (!logBox) return;
      const ts = new Date().toLocaleTimeString();
      const text = `[${ts}] ${line}`;
      const div = document.createElement("div");
      div.className = "terminal-line info";
      const upper = String(line || "").toUpperCase();
      if (upper.includes("ERROR") || upper.includes("失败")) div.className = "terminal-line error";
      else if (upper.includes("WARN") || upper.includes("警告")) div.className = "terminal-line warn";
      if (logBox.textContent.trim().includes("等待操作")) logBox.innerHTML = "";
      div.textContent = text;
      logBox.appendChild(div);
      logBox.scrollTop = logBox.scrollHeight;
    }
    function lockLog(line) {
      if (!lockLogBox) return;
      const ts = new Date().toLocaleTimeString();
      const next = `[${ts}] ${line}\n`;
      if (lockLogBox.textContent.trim() === "暂无锁定操作。") lockLogBox.textContent = "";
      lockLogBox.textContent += next;
      lockLogBox.scrollTop = lockLogBox.scrollHeight;
    }
    function setAnalyzeLockReady(ready) {
      if (!btnAnalyzeLock) return;
      btnAnalyzeLock.disabled = !ready;
      btnAnalyzeLock.classList.toggle("btn-ready", !!ready);
    }
    function makeProgressId() {
      try {
        if (crypto && crypto.randomUUID) return crypto.randomUUID();
      } catch (e) {}
      return `p_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    }
    async function fetchStrategyPlan(sid, payload) {
      const resp = await fetch("/api/strategy_plan", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ ...payload }),
      });
      const data = await resp.json();
      return { resp, data };
    }

    function openModal() {
      modal.classList.remove("hidden");
      modal.classList.add("flex");
      document.getElementById("modal-ip").focus();
    }
    function closeModal() {
      modal.classList.add("hidden");
      modal.classList.remove("flex");
      document.getElementById("modal-ip").value = "";
      document.getElementById("modal-port").value = "7777";
      document.getElementById("modal-user").value = "";
      document.getElementById("modal-pass").value = "";
    }

    function openEvalModal() {
      if (!evalModal) return;
      evalModal.classList.remove("hidden");
      evalModal.classList.add("flex");
      setEvalStep(1);
    }
    function closeEvalModal() {
      if (!evalModal) return;
      evalModal.classList.add("hidden");
      evalModal.classList.remove("flex");
    }
    let evalStep = 1;
    function currentEvalMode() {
      const el = document.querySelector('input[name="eval-mode"]:checked');
      return el ? el.value : "single";
    }
    function updateEvalModeUI() {
      const mode = currentEvalMode();
      const multiEnabled = mode === "multi";
      if (evalSingleWrap) evalSingleWrap.classList.toggle("hidden", multiEnabled);
      if (evalMultiWrap) evalMultiWrap.classList.toggle("hidden", !multiEnabled);
      if (serverSelectMulti) {
        serverSelectMulti.checked = multiEnabled;
        serverSelectMulti.dispatchEvent(new Event("change"));
      }
      if (serverSelectMultiContainer) {
        serverSelectMultiContainer.classList.toggle("hidden", !multiEnabled);
      }
      updateEvalSummary();
    }
    function setEvalStep(step) {
      evalStep = step;
      evalModal?.querySelectorAll(".eval-step").forEach((el) => {
        el.classList.toggle("hidden", Number(el.dataset.step) !== step);
      });
      evalModal?.querySelectorAll(".step-item").forEach((el) => {
        const target = Number(el.dataset.target || 0);
        el.classList.toggle("active", target > 0 && target <= step);
      });
      if (step === 2) updateEvalModeUI();
      updateEvalSummary();
    }

    function updateEvalSummary() {
      if (!evalSummary) return;
      const mode = currentEvalMode() === "multi" ? "多服务器" : "单服务器";
      let serverText = "-";
      if (currentEvalMode() === "multi") {
        const selected = Array.from(serverSelectMultiList?.querySelectorAll(".server-multi-check") || [])
          .filter((el) => el.checked)
          .map((el) => el.value);
        const modeText = (serverSelectMultiMode?.value || "distributed").toLowerCase() === "compare" ? "对比" : "分布式";
        serverText = selected.length ? `${modeText} · ${selected.join(", ")}` : `${modeText} · 未选择`;
      } else {
        serverText = serverSelect?.value || "local";
      }
      const algo = algoSelect?.value || "-";
      const dataset = evalDataset?.value || "-";
      const warm = warmupInput?.value || "0";
      if (summaryMode) summaryMode.textContent = mode;
      if (summaryServer) summaryServer.textContent = serverText;
      if (summaryAlgo) summaryAlgo.textContent = algo;
      if (summaryDataset) summaryDataset.textContent = dataset;
      if (summaryWarmup) summaryWarmup.textContent = warm;
    }

    function openLockModal() {
      lockModal.classList.remove("hidden");
      lockModal.classList.add("flex");
      updateLockModeVisibility();
      lockScope.focus();
    }
    function closeLockModal() {
      lockModal.classList.add("hidden");
      lockModal.classList.remove("flex");
    }

    btnOpenModal.addEventListener("click", openModal);
    btnCloseModal.addEventListener("click", closeModal);
    btnCancel.addEventListener("click", closeModal);
    modal.addEventListener("click", (e) => {
      if (e.target === modal) closeModal();
    });
    btnOpenEval?.addEventListener("click", openEvalModal);
    btnEvalClose?.addEventListener("click", closeEvalModal);
    evalModal?.addEventListener("click", (e) => {
      if (e.target === evalModal) closeEvalModal();
    });
    evalNext1?.addEventListener("click", () => setEvalStep(2));
    evalNext2?.addEventListener("click", () => setEvalStep(3));
    evalPrev2?.addEventListener("click", () => setEvalStep(1));
    evalPrev3?.addEventListener("click", () => setEvalStep(2));
    document.querySelectorAll('input[name="eval-mode"]').forEach((el) => {
      el.addEventListener("change", updateEvalModeUI);
    });
    evalModal?.querySelectorAll(".step-item").forEach((el) => {
      el.addEventListener("click", () => {
        const target = Number(el.dataset.target || 0);
        if (target) setEvalStep(target);
      });
    });
    btnLock?.addEventListener("click", openLockModal);
    lockClose?.addEventListener("click", closeLockModal);

    document.getElementById("btn-clear-log").addEventListener("click", () => {
      if (logBox) logBox.innerHTML = "";
    });
    btnLockLogClear?.addEventListener("click", () => {
      if (lockLogBox) lockLogBox.textContent = "";
    });

    function updateServerSelect() {
      const current = serverSelect.value;
      serverSelect.innerHTML = sources.map((s) => `<option value="${s.id}">${s.label}</option>`).join("");
      serverSelect.value = sources.some((s) => s.id === current) ? current : "local";
      updateRunGpuOptions();
      updateEvalSummary();
    }

    function updateServerMultiList() {
      if (!serverSelectMultiList) return;
      serverSelectMultiList.innerHTML = sources
        .map((s) => {
          const snap = snapshots.get(s.id) || {};
          const gpus = Array.isArray(snap.gpus) ? snap.gpus : [];
          const active = (snap.error ? false : true);
          return `
            <div class="server-multi-item">
              <label class="server-multi-label">
                <input type="checkbox" class="server-multi-check" value="${s.id}" />
                <div class="server-multi-info">
                  <div class="server-multi-name">${s.label}</div>
                  <div class="server-multi-sub">GPU ${gpus.length} 张 · ${snap.hostname || "N/A"}</div>
                </div>
                <span class="server-tag ${active ? "active" : ""}">${active ? "在线" : "未知"}</span>
              </label>
            </div>
          `;
        })
        .join("");
    }

    function updateLockScopeOptions() {
      if (!lockScope) return;
      const opts = [{ id: "all", label: "所有服务器" }, ...sources.map((s) => ({ id: s.id, label: s.label }))];
      const current = lockScope.value;
      lockScope.innerHTML = opts.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
      lockScope.value = opts.some((o) => o.id === current) ? current : "all";
      if (lockStatusScope) {
        const cur = lockStatusScope.value;
        lockStatusScope.innerHTML = opts.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
        lockStatusScope.value = opts.some((o) => o.id === cur) ? cur : "all";
      }
    }

    function updateLockManualOptions() {
      if (!lockServerList) return;
      const scope = lockScope?.value || "all";
      const targets = scope === "all" ? sources : sources.filter((s) => s.id === scope);
      if (!targets.length) {
        lockServerList.innerHTML = `<div style="font-size:12px;color:var(--muted);">未找到可用服务器</div>`;
        return;
      }
      lockServerList.innerHTML = targets
        .map((s) => {
          const snap = snapshots.get(s.id) || {};
          const gpus = Array.isArray(snap.gpus) ? snap.gpus : [];
          const gpuHtml = gpus.length
            ? gpus
                .map((g) => {
                  const label = g.name ? `${g.name} (#${g.id})` : `GPU ${g.id}`;
                  return `
                    <label style="display:flex;gap:10px;align-items:center;border:1px solid var(--card-border);border-radius:10px;padding:8px;background:#fff;">
                      <input type="checkbox" class="lock-gpu-check" data-server="${s.id}" value="${g.id}" />
                      <span style="font-weight:900;">${label}</span>
                    </label>
                  `;
                })
                .join("")
            : `<div style="font-size:12px;color:var(--muted);">无 GPU</div>`;
          return `
            <div style="border:1px solid var(--card-border);border-radius:10px;padding:10px;background:#f8fafc;">
              <div style="font-weight:900;margin-bottom:8px;">${s.label}</div>
              <div style="display:grid;gap:8px;">${gpuHtml}</div>
            </div>
          `;
        })
        .join("");
    }

    function updateLockModeVisibility() {
      const mode = (lockMode?.value || "auto").toLowerCase();
      const isManual = mode === "manual";
      lockWarmupWrap?.classList.toggle("hidden", isManual);
      lockManualWrap?.classList.toggle("hidden", !isManual);
      if (isManual) updateLockManualOptions();
    }

    function formatExpire(ts) {
      if (!ts) return "-";
      try {
        const d = new Date(ts * 1000);
        return d.toLocaleString();
      } catch (e) {
        return "-";
      }
    }

    function renderLockStatus(data) {
      if (!lockStatusBox) return;
      const results = (data && data.results) || {};
      const entries = Object.entries(results);
      if (!entries.length) {
        lockStatusBox.innerHTML = `<div style="font-size:12px;color:var(--muted);">暂无锁定状态。</div>`;
        return;
      }
      lockStatusBox.innerHTML = entries
        .map(([sid, info]) => {
          if (info?.error) {
            return `<div class="stat" style="border-color:#fecaca;background:#fef2f2;">
              <div class="stat-label">${sid}</div>
              <div class="stat-value">错误：${info.error}</div>
            </div>`;
          }
          const active = info?.active ? "已锁定" : "未锁定";
          const backend = info?.backend || "-";
          const devices = Array.isArray(info?.devices) ? info.devices.join(",") : "-";
          const exp = formatExpire(info?.expires_at || 0);
          const mem = info?.mem_mb != null ? `${info.mem_mb} MB` : "-";
          const warm = info?.warmup_iters != null ? `${info.warmup_iters}` : "-";
          return `
            <div class="stat" style="min-height:auto;">
              <div class="stat-label">${sid}</div>
              <div class="stat-value">${active}</div>
              <div style="font-size:11px;color:var(--muted);margin-top:4px;">
                backend=${backend} devices=${devices} expires=${exp} mem=${mem} warmup=${warm}
                ${info?.note === "no_available_gpu" ? " · 无空闲GPU，已跳过锁定" : ""}
              </div>
            </div>
          `;
        })
        .join("");
    }

    async function fetchLockStatus(scope) {
      const target = scope || (lockStatusScope?.value || "all");
      try {
        const resp = await fetch(`/api/resource_lock/status?scope=${encodeURIComponent(target)}`, { cache: "no-store" });
        const data = await resp.json();
        if (!resp.ok) {
          lockLog(`锁定状态获取失败: ${JSON.stringify(data)}`);
          return;
        }
        renderLockStatus(data);
      } catch (e) {
        lockLog(`锁定状态获取失败: ${e}`);
      }
    }

    function plansLoad() {
      try { return JSON.parse(localStorage.getItem(PLAN_KEY) || "{}") || {}; } catch (e) { return {}; }
    }
    function plansSave(obj) {
      try { localStorage.setItem(PLAN_KEY, JSON.stringify(obj)); } catch (e) {}
    }
    function getSavedPlan(server_id) {
      const all = plansLoad();
      return all[server_id] || null;
    }
    function setSavedPlan(server_id, plan) {
      const all = plansLoad();
      all[server_id] = { plan, saved_at: new Date().toLocaleString() };
      plansSave(all);
    }

    function updateRunGpuOptions() {
      const sid = RUN_SERVER_ID;
      if (sid === "all") {
        if (runGpuSingle) runGpuSingle.innerHTML = "";
        if (runGpuMulti) runGpuMulti.innerHTML = "";
        return;
      }
      const snap = snapshots.get(sid) || {};
      const gpus = Array.isArray(snap.gpus) ? snap.gpus : [];
      const options = gpus.map((g) => ({ id: g.id, name: g.name || `GPU ${g.id}` }));

      if (runGpuSingle) {
        runGpuSingle.innerHTML = options.map((o) => `<option value="${o.id}">${o.name} (#${o.id})</option>`).join("");
      }
      if (runGpuMulti) {
        runGpuMulti.innerHTML = options
          .map((o) => {
            return `
              <label style="display:flex;gap:10px;align-items:center;border:1px solid var(--card-border);border-radius:10px;padding:8px;background:#fff;">
                <input type="checkbox" class="run-gpu-check" value="${o.id}" />
                <span style="font-weight:900;">${o.name}</span>
                <span style="color:var(--muted);font-size:12px;">#${o.id}</span>
              </label>
            `;
          })
          .join("");
      }

      // Apply default selection from saved plan for AUTO
      const saved = getSavedPlan(sid);
      if (saved && saved.plan && Array.isArray(saved.plan.devices) && saved.plan.devices.length) {
        const devs = saved.plan.devices.map((x) => Number(x));
        if (runGpuSingle && devs.length) runGpuSingle.value = String(devs[0]);
        if (runGpuMulti) {
          runGpuMulti.querySelectorAll(".run-gpu-check").forEach((el) => {
            el.checked = devs.includes(Number(el.value));
          });
        }
      }
    }

    function updateRunServerOptions(lockResults) {
      if (!runServer) return;
      const results = lockResults || {};
      const locked = Object.keys(results).filter((k) => results[k]?.active);
      const remoteLocked = locked.filter((k) => k !== "local");
      const opts = [];
      opts.push({ id: "all", label: "默认（所有服务器）" });
      if (results.local?.active || locked.length === 0) {
        opts.push({ id: "local", label: (sources.find((s) => s.id === "local") || {}).label || "本机" });
      }
      remoteLocked.forEach((sid) => {
        const src = sources.find((s) => s.id === sid);
        opts.push({ id: sid, label: src?.label || sid });
      });
      const current = runServer.value || RUN_SERVER_ID;
      runServer.innerHTML = opts.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
      runServer.value = opts.some((o) => o.id === current) ? current : opts[0].id;
      RUN_SERVER_ID = runServer.value;
      updateRunModeOptions();
      updateRunGpuOptions();
      updateRunModeVisibility();
    }

    function updateRunModeOptions() {
      if (!runMode) return;
      const sid = runServer?.value || RUN_SERVER_ID;
      const snap = snapshots.get(sid) || {};
      const hasGpu = Array.isArray(snap.gpus) && snap.gpus.length > 0;
      let opts = [];
      if (sid === "all") {
        opts = [
          { id: "AUTO", label: "默认（使用资源评估结果）" },
          { id: "MNM", label: "MNM（多机适应度加速）" },
        ];
      } else if (!hasGpu) {
        opts = [{ id: "S", label: "S（单卡）" }];
      } else {
        opts = [
          { id: "S", label: "S（单卡）" },
          { id: "SM", label: "SM（单卡加速）" },
          { id: "M", label: "M（多卡）" },
        ];
      }
      const current = runMode.value || "AUTO";
      runMode.innerHTML = opts.map((o) => `<option value="${o.id}">${o.label}</option>`).join("");
      runMode.value = opts.some((o) => o.id === current) ? current : opts[0].id;
    }

    function updateRunModeVisibility() {
      const mode = (runMode?.value || "AUTO").toUpperCase();
      const sid = RUN_SERVER_ID;
      if (sid === "all") {
        runGpuSingleWrap?.classList.add("hidden");
        runGpuMultiWrap?.classList.add("hidden");
        return;
      }
      const snap = snapshots.get(sid) || {};
      const hasGpu = Array.isArray(snap.gpus) && snap.gpus.length > 0;

      const single = mode === "S" || mode === "SM";
      const multi = mode === "M" || mode === "MNM";
      runGpuSingleWrap?.classList.toggle("hidden", !(single && hasGpu));
      runGpuMultiWrap?.classList.toggle("hidden", !(multi && hasGpu));
    }
    runMode?.addEventListener("change", updateRunModeVisibility);
    runMode?.addEventListener("change", updateRunModeVisibility);
    runServer?.addEventListener("change", () => {
      RUN_SERVER_ID = runServer.value;
      updateRunModeOptions();
      updateRunGpuOptions();
      updateRunModeVisibility();
      refreshRunLockStatus();
      saveRunConfig();
    });

    function saveRunConfig() {
      if (!runAlgo) return;
      try {
        const multi = runGpuMulti
          ? Array.from(runGpuMulti.querySelectorAll(".run-gpu-check") || [])
              .filter((el) => el.checked)
              .map((el) => String(el.value))
          : [];
        const payload = {
          runAlgo: runAlgo.value,
          runDataset: runDataset?.value || "",
          runServer: runServer?.value || RUN_SERVER_ID,
          runMode: runMode?.value || "AUTO",
          runGpuSingle: runGpuSingle?.value || "",
          runGpuMulti: multi,
          runIters: runIters?.value || "",
          runPc: runPc?.value || "",
          runPm: runPm?.value || "",
        };
        localStorage.setItem(RUN_CONFIG_KEY, JSON.stringify(payload));
      } catch (e) {}
    }

    function applyRunConfig() {
      if (!runAlgo) return;
      let cfg = null;
      try {
        cfg = JSON.parse(localStorage.getItem(RUN_CONFIG_KEY) || "null");
      } catch (e) {
        cfg = null;
      }
      if (!cfg) return;
      if (cfg.runAlgo && runAlgo.querySelector(`option[value="${cfg.runAlgo}"]`)) {
        runAlgo.value = cfg.runAlgo;
      }
      updateRunDatasetOptions();
      if (cfg.runDataset && runDataset?.querySelector(`option[value="${cfg.runDataset}"]`)) {
        runDataset.value = cfg.runDataset;
      }
      if (cfg.runServer && runServer?.querySelector(`option[value="${cfg.runServer}"]`)) {
        runServer.value = cfg.runServer;
        RUN_SERVER_ID = cfg.runServer;
      }
      updateRunModeOptions();
      if (cfg.runMode && runMode?.querySelector(`option[value="${cfg.runMode}"]`)) {
        runMode.value = cfg.runMode;
      }
      updateRunGpuOptions();
      updateRunModeVisibility();
      if (cfg.runGpuSingle && runGpuSingle?.querySelector(`option[value="${cfg.runGpuSingle}"]`)) {
        runGpuSingle.value = cfg.runGpuSingle;
      }
      if (Array.isArray(cfg.runGpuMulti) && runGpuMulti) {
        const want = new Set(cfg.runGpuMulti.map((v) => String(v)));
        runGpuMulti.querySelectorAll(".run-gpu-check").forEach((el) => {
          el.checked = want.has(String(el.value));
        });
      }
      if (cfg.runIters && runIters) runIters.value = cfg.runIters;
      if (cfg.runPc && runPc) runPc.value = cfg.runPc;
      if (cfg.runPm && runPm) runPm.value = cfg.runPm;
    }

    runAlgo?.addEventListener("change", () => {
      updateRunDatasetOptions();
      saveRunConfig();
    });
    runDataset?.addEventListener("change", saveRunConfig);
    runMode?.addEventListener("change", saveRunConfig);
    runGpuSingle?.addEventListener("change", saveRunConfig);
    runGpuMulti?.addEventListener("change", saveRunConfig);
    runIters?.addEventListener("change", saveRunConfig);
    runPc?.addEventListener("change", saveRunConfig);
    runPm?.addEventListener("change", saveRunConfig);

    btnAdd.addEventListener("click", () => {
      const ip = document.getElementById("modal-ip").value.trim();
      const port = document.getElementById("modal-port").value.trim() || "7777";
      const username = document.getElementById("modal-user").value.trim();
      const password = document.getElementById("modal-pass").value;
      if (!ip) return alert("请填写IP地址");
      const base = `http://${ip}:${port}`;
      const id = `${ip}-${port}`;
      if (sources.some((s) => s.id === id)) return alert("已添加该服务器");
      sources.push({ id, base, label: `${ip}:${port}`, expanded: false, meta: { type: "remote", username, password } });
      updateServerSelect();
      updateLockScopeOptions();
      closeModal();
      log(`已添加服务器 ${id}`);

      // Immediate feedback for the newly added server
      const added = sources.find((s) => s.id === id);
      if (added) {
        fetchFrom(added).then((data) => {
          snapshots.set(id, data || {});
          if (data?.error) {
            log(`服务器 ${id} 连接失败：${data.error}`);
            log(`请确认远程已启动 server_agent，并监听 ${port}（以及防火墙放通）。`);
          }
          render();
        });
      }
      refreshAll();
    });

    async function loadConfigServers() {
      try {
        const res = await fetch("/api/servers", { cache: "no-store" });
        if (!res.ok) return;
        const list = await res.json();
        list.forEach((item) => {
          if (!item.ip) return;
          const id = item.id || `${item.ip}-${item.port || "default"}`;
          if (sources.some((s) => s.id === id)) return;
          const protocol = item.protocol || "http";
          const base = `${protocol}://${item.ip}${item.port ? ":" + item.port : ""}`;
          sources.push({ id, base, label: item.name || `${item.ip}:${item.port || ""}`, expanded: false, meta: { type: item.type, username: item.username } });
        });
        updateServerSelect();
        updateLockScopeOptions();
      } catch (e) {
        console.warn("load servers failed", e);
      }
    }

    lockMode?.addEventListener("change", updateLockModeVisibility);
    lockScope?.addEventListener("change", () => {
      updateLockManualOptions();
      updateLockModeVisibility();
    });
    serverSelectMulti?.addEventListener("change", () => {
      const enabled = !!serverSelectMulti.checked;
      serverSelectMultiContainer?.classList.toggle("hidden", !enabled);
      serverSelectMultiList?.classList.toggle("hidden", !enabled);
      serverSelectMultiMode?.classList.toggle("hidden", !enabled);
      if (enabled) updateServerMultiList();
      if (serverSelect) serverSelect.disabled = enabled;
      updateEvalSummary();
    });

    async function loadAlgorithms() {
      try {
        const res = await fetch("/api/algorithms", { cache: "no-store" });
        if (!res.ok) return;
        const list = await res.json();
        const opts = [{ id: "monitor", name: "仅监控资源", kind: "monitor" }, ...(Array.isArray(list) ? list : [])];
        algoSelect.innerHTML = opts
          .map((o) => {
            const disabled = o.disabled ? "disabled" : "";
            const label = o.name || o.id;
            const kind = o.kind || "ga";
            return `<option value="${o.id}" data-kind="${kind}" ${disabled}>${label}</option>`;
          })
          .join("");
        if (runAlgo) {
          const runOpts = opts.filter((o) => o.id !== "monitor");
          runAlgo.innerHTML = runOpts
            .map((o) => {
              const disabled = o.disabled ? "disabled" : "";
              const label = o.name || o.id;
              return `<option value="${o.id}" ${disabled}>${label}</option>`;
            })
            .join("");
        }
      } catch (e) {
        console.warn("load algorithms failed", e);
      }
    }

    async function loadDatasets() {
      try {
        const res = await fetch("/api/datasets", { cache: "no-store" });
        if (!res.ok) return;
        datasetsManifest = await res.json();
      } catch (e) {
        datasetsManifest = null;
      }
    }

    function updateRunDatasetOptions() {
      if (!runDataset) return;
      const algo = runAlgo?.value;
      const byAlgo = datasetsManifest?.by_algorithm || {};
      const list = Array.isArray(byAlgo?.[algo]) ? byAlgo[algo] : [];
      if (!list.length) {
        runDataset.innerHTML = `<option value="">(该算法暂无数据集清单)</option>`;
        return;
      }
      runDataset.innerHTML = list.map((d) => `<option value="${d}">${d}</option>`).join("");
    }

    function updateEvalDatasetOptions() {
      if (!evalDataset) return;
      const algo = algoSelect?.value;
      const byAlgo = datasetsManifest?.by_algorithm || {};
      const list = Array.isArray(byAlgo?.[algo]) ? byAlgo[algo] : [];
      if (!list.length) {
        evalDataset.innerHTML = `<option value="">(该算法暂无数据集清单)</option>`;
        return;
      }
      evalDataset.innerHTML = list.map((d) => `<option value="${d}">${d}</option>`).join("");
    }

    algoSelect.addEventListener("change", () => {
      const sel = algoSelect.selectedOptions[0];
      const kind = sel?.dataset?.kind || (algoSelect.value === "monitor" ? "monitor" : "ga");
      warmupWrap.classList.toggle("hidden", kind === "monitor");
      if (evalDataset) evalDataset.disabled = kind === "monitor";
      updateEvalDatasetOptions();
      updateEvalSummary();
    });

    runAlgo?.addEventListener("change", updateRunDatasetOptions);
    serverSelect?.addEventListener("change", updateEvalSummary);
    serverSelectMultiMode?.addEventListener("change", updateEvalSummary);
    serverSelectMultiList?.addEventListener("change", (event) => {
      const target = event.target;
      if (target && target.classList && target.classList.contains("server-multi-check")) {
        updateEvalSummary();
      }
    });
    evalDataset?.addEventListener("change", updateEvalSummary);
    warmupInput?.addEventListener("change", updateEvalSummary);

    async function fetchFrom(source) {
      try {
        const url = source.base ? `${source.base}/api/resources` : "/api/resources";
        const res = await fetch(url, { cache: "no-store" });
        if (!res.ok) throw new Error(res.status);
        return await res.json();
      } catch (e) {
        return { error: "连接失败或无响应", hostname: source.label };
      }
    }

    function render() {
      cardsDiv.innerHTML = sources
        .map((src) => {
          const data = snapshots.get(src.id) || {};
          const gpus = data.gpus || [];
          const cpu = data.cpu || {};
          const mem = data.memory || {};
          const expanded = !!src.expanded;

          const title = data.name || src.label || data.hostname || src.id;
          const gpuHtml = gpus.length
            ? gpus
                .map(
                  (g) => `
                  <div class="gpu-item">
                    <div class="gpu-title">GPU ${g.id} · ${g.name}</div>
                    <div style="color:var(--muted);margin-top:2px;">
                      显存：${fmt(g.used_mb)} / ${fmt(g.total_mb)} MB（功率 ${fmt(g.power_w, " W")}）
                    </div>
                    <div style="color:var(--muted);margin-top:2px;">
                      GPU 利用率：${fmt(g.gpu_util_percent, "%")} · 显存利用率：${fmt(g.mem_util_percent, "%")}
                    </div>
                  </div>`
                )
                .join("")
            : `<div style="font-size:12px;color:var(--muted);margin-top:6px;">无GPU信息</div>`;

          return `
            <div class="card ${expanded ? "card--expanded" : ""}" onclick="toggle('${src.id}')">
              <div class="card-top">
                <div class="card-left">
                  <div class="card-head">
                    <div>
                      <div class="card-name">${title}</div>
                      <div class="card-type">${src.meta?.type === "local" ? "本地机器" : src.meta?.type || "远程服务器"}</div>
                    </div>
                    <div class="badge">${data.time || data.timestamp || "--"}</div>
                  </div>
                  <div style="font-size:12px;color:var(--muted);">
                    Host: ${data.hostname || "N/A"}
                  </div>
                </div>

                <div class="card-mid">
                  <div class="stats">
                    <div class="stat">
                      <div class="stat-label">CPU 负载</div>
                      <div class="stat-value">${pct(cpu.usage_percent)}</div>
                    </div>
                    <div class="stat">
                      <div class="stat-label">内存占用</div>
                      <div class="stat-value">${pct(mem.percent)}</div>
                    </div>
                    <div class="stat">
                      <div class="stat-label">显卡数量</div>
                      <div class="stat-value">${gpus.length} 张</div>
                    </div>
                  </div>
                  ${data.error ? `<div class="error">错误：${data.error}</div>` : ""}
                </div>
              </div>

              <div class="card-bottom">
                <div class="details">
                  <div>CPU 负载均值：${fmt(cpu.load_avg?.[0], "", "N/A")}</div>
                  <div>内存：${fmt(mem.used_mb)} / ${fmt(mem.total_mb)} MB</div>
                  ${src.meta?.username ? `<div>用户：${src.meta.username}</div>` : ""}
                  ${gpuHtml}
                </div>
              </div>
            </div>
          `;
        })
        .join("");
    }

    function toggle(id) {
      const s = sources.find((x) => x.id === id);
      if (s) s.expanded = !s.expanded;
      render();
    }

    async function refreshAll() {
      try {
        const resp = await fetch("/api/all_resources", { cache: "no-store" });
        const all = await resp.json();
        Object.entries(all).forEach(([id, data]) => snapshots.set(id, data || {}));
        render();
        updateRunGpuOptions();
        updateRunModeOptions();
        updateRunModeVisibility();
      } catch (e) {
        console.error("获取资源失败", e);
      }
    }

    document.getElementById("btn-refresh").addEventListener("click", refreshAll);

    // Analysis interactions
    document.getElementById("btn-analyze").addEventListener("click", async () => {
      lastEvalPlan = null;
      lastEvalMode = null;
      lastEvalTarget = null;
      setAnalyzeLockReady(false);
      log("评估任务准备中...");
      try {
      const algo = algoSelect.value;
      const warmup = Math.max(0, Number(warmupInput.value || 0));
      const tpeTrials = Math.max(1, Number(tpeTrialsInput?.value || 6));
      const tpeWarmup = Math.max(1, Number(tpeWarmupInput?.value || 1));
      const gpuBusyThreshold = Math.max(0, Math.min(100, Number(gpuBusyInput?.value || 60)));
      const minGpuFree = Math.max(0, Number(gpuFreeInput?.value || 1024));
      log(`评估参数：tpe_trials=${tpeTrials} tpe_warmup=${tpeWarmup} gpu_busy<=${gpuBusyThreshold}% free>=${minGpuFree}MB`);
      const isMonitor = algo === "monitor";
      const allowWarmup = !isMonitor && warmup > 0 && !(serverSelectMulti?.checked && (serverSelectMultiMode?.value || "distributed").toLowerCase() === "compare");
      const multiMode = !!serverSelectMulti?.checked;
      const multiPurpose = (serverSelectMultiMode?.value || "distributed").toLowerCase();
      const multiTargets = Array.from(serverSelectMultiList?.querySelectorAll(".server-multi-check") || [])
        .filter((el) => el.checked)
        .map((el) => el.value);
      const useMulti = multiMode && multiTargets.length > 0;
      const targets = useMulti ? multiTargets : [serverSelect.value];
      if (multiMode && !useMulti) {
        log("未选择多服务器，已回退单服务器评估。");
      }

      let panelPinned = false;
      function renderComparison(compare, selectedPlan) {
        if (!compare || !Array.isArray(compare.candidates)) return;
        panelPinned = true;
        const best = compare.best || {};
        const selected = selectedPlan || {};
        const items = compare.candidates
          .map((c) => c?.plan)
          .filter((p) => p && typeof p.estimated_time_ms === "number");
        if (!items.length) return;

        const maxMs = Math.max(...items.map((p) => p.estimated_time_ms || 0), 1);
        const rows = compare.candidates
          .map((c) => {
            const p = c.plan || {};
            const ms = typeof p.estimated_time_ms === "number" ? p.estimated_time_ms : null;
            if (ms == null) return null;
            let label = c.tag || p.backend || "plan";
            if (p.backend === "cpu") label = "CPU";
            if (p.backend === "cuda") label = `GPU(${(p.devices || [])[0] ?? "-"})`;
            if (p.backend === "multi-gpu") label = `Multi-GPU(${(p.devices || []).length})`;
            const pct = Math.max(2, Math.round((ms / maxMs) * 100));
            const isSelected = (p.backend === selected.backend) && JSON.stringify(p.devices || []) === JSON.stringify(selected.devices || []);
            const speedup = ms > 0 && best.estimated_time_ms ? (ms / best.estimated_time_ms) : null;
            return { label, ms, pct, isSelected, speedup };
          })
          .filter(Boolean);

        const bestMs = typeof best.estimated_time_ms === "number" ? best.estimated_time_ms : null;
        const selectedMs = typeof selected.estimated_time_ms === "number" ? selected.estimated_time_ms : null;
        const baseMs = selectedMs || bestMs;
        const bestLabel = best.backend === "cpu"
          ? "CPU"
          : best.backend === "cuda"
            ? `GPU(${(best.devices || [])[0] ?? "-"})`
            : best.backend === "multi-gpu"
              ? `Multi-GPU(${(best.devices || []).length})`
              : (best.backend || "best");
        const selectedLabel = selected.backend === "cpu"
          ? "CPU"
          : selected.backend === "cuda"
            ? `GPU(${(selected.devices || [])[0] ?? "-"})`
            : selected.backend === "multi-gpu"
              ? `Multi-GPU(${(selected.devices || []).length})`
              : (selected.backend || "策略");

        const barsHtml = rows
          .map((r) => {
            const color = r.isSelected ? "#16a34a" : "#94a3b8";
            const border = r.isSelected ? "1px solid #86efac" : "1px solid #e2e8f0";
            const bg = r.isSelected ? "#ecfdf5" : "#f8fafc";
            let hint = r.isSelected ? "当前策略" : "";
            if (!r.isSelected && baseMs) {
              hint = r.ms <= baseMs ? `快 ${(baseMs / r.ms).toFixed(2)}×` : `慢 ${(r.ms / baseMs).toFixed(2)}×`;
            }
            return `
              <div style="display:grid;gap:6px;padding:10px;border-radius:10px;background:${bg};border:${border};">
                <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
                  <div style="font-weight:900;">${r.label}</div>
                  <div style="font-size:12px;color:var(--muted);white-space:nowrap;">${r.ms.toFixed(2)} ms ${hint ? `· ${hint}` : ""}</div>
                </div>
                <div style="height:10px;background:#e2e8f0;border-radius:999px;overflow:hidden;">
                  <div style="width:${r.pct}%;height:100%;background:${color};"></div>
                </div>
              </div>
            `;
          })
          .join("");

        gaPanel.innerHTML = `
          <div style="width:100%;display:grid;gap:12px;">
            <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
              <div style="font-weight:900;">静态候选对比</div>
              <div style="font-size:12px;color:var(--muted);">当前策略：${selectedLabel}${selected.estimated_time_ms != null ? ` · ${Number(selected.estimated_time_ms).toFixed(2)} ms` : ""}</div>
            </div>
            <div style="font-size:11px;color:var(--muted);">说明：此处为静态候选对比，最终策略以 StrategyPlan 选择为准。</div>
            <div style="display:grid;grid-template-columns:1fr;gap:10px;">
              ${barsHtml}
            </div>
          </div>
        `;
      }

      function planToMode(plan) {
        if (!plan) return { mode: "AUTO", devices: null };
        const backend = plan.backend || (plan.plan || {}).backend;
        const devices = plan.devices || (plan.plan || {}).devices || [];
        if (!backend) return { mode: "AUTO", devices };
        if (backend === "cpu") return { mode: "CPU", devices: [] };
        if (backend === "cuda") return { mode: "S", devices };
        if (backend === "multi-gpu") return { mode: "M", devices };
        return { mode: "AUTO", devices };
      }

      function renderWarmupSummary(title, data) {
        if (!data) return;
        const state = data.state || "unknown";
        const summary = data.summary || {};
        const perIter = Array.isArray(data.per_iter_ms) ? data.per_iter_ms : [];
        const comm = data.comm || {};
        const iterMs = summary.iter_avg_ms;
        const iterSec = summary.iter_seconds;
        const throughput = summary.throughput_ips;
        const mode = summary.mode || "-";
        const devs = Array.isArray(summary.devices) ? summary.devices.join(",") : (summary.devices ?? "-");
        const remote = Array.isArray(summary.remote_servers) ? summary.remote_servers.join(", ") : "";

        const w = 520, h = 120, pad = 14;
        let spark = "";
        if (perIter.length >= 2) {
          const minV = Math.min(...perIter);
          const maxV = Math.max(...perIter);
          const points = perIter
            .map((y, i) => {
              const x = pad + (i / (perIter.length - 1)) * (w - pad * 2);
              const yy = pad + (1 - (y - minV) / Math.max(1e-9, (maxV - minV))) * (h - pad * 2);
              return `${x.toFixed(1)},${yy.toFixed(1)}`;
            })
            .join(" ");
          spark = `
            <div style="border:1px dashed var(--card-border);border-radius:10px;background:var(--soft-bg);padding:8px;">
              <svg viewBox="0 0 ${w} ${h}" width="100%" height="120" preserveAspectRatio="none">
                <polyline fill="none" stroke="#2563eb" stroke-width="3" points="${points}" />
                <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#cbd5e1" stroke-width="2"/>
                <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="#cbd5e1" stroke-width="2"/>
              </svg>
            </div>
          `;
        }

        const commAvg = comm.avg_ms ? `${Number(comm.avg_ms).toFixed(2)} ms` : "-";
        const stateText = state === "completed" ? "完成" : state;

        gaPanel.insertAdjacentHTML(
          "beforeend",
          `
            <div style="display:grid;gap:10px;padding:12px;border-radius:12px;border:1px solid var(--card-border);background:#fff;">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
                <div style="font-weight:900;">Warmup 实测（${title}）</div>
                <div style="font-size:12px;color:var(--muted);">状态：${stateText}</div>
              </div>
              <div style="display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:8px;">
                <div style="font-size:12px;color:var(--muted);">模式：<span style="color:var(--ink);font-weight:700;">${mode}</span></div>
                <div style="font-size:12px;color:var(--muted);">设备：<span style="color:var(--ink);font-weight:700;">${devs}</span></div>
                <div style="font-size:12px;color:var(--muted);">迭代均时：<span style="color:var(--ink);font-weight:700;">${iterMs ? Number(iterMs).toFixed(2) + " ms" : "-"}</span></div>
                <div style="font-size:12px;color:var(--muted);">吞吐：<span style="color:var(--ink);font-weight:700;">${throughput ? Number(throughput).toFixed(2) + " ind/s" : "-"}</span></div>
                <div style="font-size:12px;color:var(--muted);">迭代总时：<span style="color:var(--ink);font-weight:700;">${iterSec ? Number(iterSec).toFixed(3) + " s" : "-"}</span></div>
                <div style="font-size:12px;color:var(--muted);">通信均时：<span style="color:var(--ink);font-weight:700;">${commAvg}</span></div>
              </div>
              ${remote ? `<div style="font-size:12px;color:var(--muted);">分布式节点：${remote}</div>` : ""}
              ${spark}
            </div>
          `
        );
      }

      async function runWarmupOnServer({ sid, plan, label, remoteServers }) {
        if (isMonitor || warmup <= 0) return null;
        const ds = evalDataset?.value || "";
        if (!ds) {
          log("未选择数据集，跳过 Warmup。");
          return null;
        }
        const base = planToMode(plan);
        const mode = remoteServers && remoteServers.length ? "MNM" : base.mode;
        const devices = base.devices;
        log(`Warmup 开始：${label} · algo=${algo} dataset=${ds} mode=${mode} iters=${warmup}`);
        try {
          const resp = await fetch("/api/ga_warmup", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
              server_id: sid,
              algorithm: algo,
              dataset: ds,
              iterations: warmup,
              mode,
              devices,
              remote_servers: remoteServers,
              timeout_s: Math.max(600, warmup * 10),
            }),
          });
          const data = await resp.json();
          if (!resp.ok) {
            log(`Warmup 失败：HTTP ${resp.status} ${JSON.stringify(data)}`);
            return null;
          }
          const summary = data.summary || {};
          const commAvg = data.comm?.avg_ms != null ? Number(data.comm.avg_ms).toFixed(2) + " ms" : "-";
          if (summary.iter_avg_ms != null || summary.throughput_ips != null) {
            const iterAvg = summary.iter_avg_ms != null ? Number(summary.iter_avg_ms).toFixed(2) + " ms" : "-";
            const throughput = summary.throughput_ips != null ? Number(summary.throughput_ips).toFixed(2) + " ind/s" : "-";
            log(`Warmup 结果：iter_avg=${iterAvg} throughput=${throughput} comm_avg=${commAvg}`);
          }
          renderWarmupSummary(label, data);
          return data;
        } catch (e) {
          log(`Warmup 调用异常：${e}`);
          return null;
        }
      }


      async function analyzeOne(sid) {
        const src = sources.find((s) => s.id === sid) || sources[0];
        if (isMonitor) {
          if (!(useMulti && multiPurpose === "distributed")) {
            log(`开始静态资源分析：${src.label}`);
          }
        } else {
          log(`开始算法分析：${src.label} · ${algo}`);
          gaPanel.innerHTML = `<div>算法分析中...</div>`;
        }

        // Ask backend for adaptive plan (StrategyPlan)
        let plan = null;
        try {
          const { resp, data } = await fetchStrategyPlan(sid, {
            server_id: sid,
            algorithm: algo,
            warmup: 0,
            objective: "time",
            multi_gpu: true,
            timeout_s: 600,
            tpe_trials: tpeTrials,
            tpe_warmup: tpeWarmup,
            gpu_busy_threshold: gpuBusyThreshold,
            min_gpu_free_mb: minGpuFree,
          });
          if (resp.ok) {
            plan = data;
          } else {
            log(`StrategyPlan 调用失败：HTTP ${resp.status} ${JSON.stringify(data)}`);
          }
        } catch (e) {
          plan = null;
          log(`StrategyPlan 调用异常：${e}`);
        }

        if (plan) {
          const planDevices = Array.isArray(plan.devices) ? plan.devices : [];
          const deviceText = plan.backend === "cuda"
            ? `GPU ${planDevices.join(",") || "-"}`
            : plan.backend === "multi-gpu"
              ? `GPU ${planDevices.join(",") || "-"}`
              : "CPU";
          log(`StrategyPlan 结果：backend=${plan.backend} devices=${planDevices.join(",") || "-"} world_size=${plan.world_size || 1}`);
          log(`选中设备：${deviceText}`);
          if (plan.estimated_time_ms != null) log(`估计耗时：${plan.estimated_time_ms.toFixed(3)} ms`);
          if (plan.reason) log(`原因：${plan.reason}`);
          try { setSavedPlan(sid, plan); } catch (e) {}
        } else {
          log(`StrategyPlan 未获取到结果，使用默认策略。`);
        }

        let compare = null;
        if (!(useMulti && multiPurpose === "distributed")) {
          try {
            const resp = await fetch("/api/strategy_compare", {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                server_id: sid,
                objective: "time",
                multi_gpu: true,
                warmup_iters: 0,
                timeout_s: 60,
                gpu_busy_threshold: gpuBusyThreshold,
                min_gpu_free_mb: minGpuFree,
              }),
            });
            if (resp.ok) {
              compare = await resp.json();
              const rows = (compare.candidates || [])
                .map((c) => c.plan)
                .filter((p) => p && typeof p.estimated_time_ms === "number")
                .map((p) => {
                  const backend = p.backend || "plan";
                  const devs = Array.isArray(p.devices) && p.devices.length ? `(${p.devices.join(",")})` : "";
                  return { label: `${backend}${devs}`, ms: Number(p.estimated_time_ms) };
                })
                .sort((a, b) => a.ms - b.ms)
                .slice(0, 4);
              if (rows.length) {
                log(`静态候选：${rows.map((r) => `${r.label}=${r.ms.toFixed(2)}ms`).join(" · ")}`);
              }
              if (plan && plan.estimated_time_ms != null) {
                const devices = Array.isArray(plan.devices) ? plan.devices.join(",") : "-";
                log(`当前策略：${plan.backend}(${devices || "-"})=${Number(plan.estimated_time_ms).toFixed(2)}ms（以 StrategyPlan 为准）`);
              } else {
                log("当前策略：以 StrategyPlan 为准");
              }
              renderComparison(compare, plan || {});
            } else {
              let errBody = null;
              try { errBody = await resp.json(); } catch (e) { errBody = { raw: await resp.text() }; }
              log(`StrategyCompare 失败：HTTP ${resp.status} ${JSON.stringify(errBody)}`);
            }
          } catch (e) {
            log(`StrategyCompare 异常：${e}`);
          }
        }

        if (isMonitor) {
          const snap = await fetchFrom(src);
          const cpu = snap.cpu || {};
          const mem = snap.memory || {};
          log(`Host: ${snap.hostname || src.label}`);
          log(`CPU: ${pct(cpu.usage_percent)} (load=${fmt(cpu.load_avg?.[0], "", "N/A")})`);
          log(`MEM: ${pct(mem.percent)} (${fmt(mem.used_mb)} / ${fmt(mem.total_mb)} MB)`);
          if (!compare) {
            gaPanel.textContent = "等待数据...";
          }
          return { sid, plan, compare };
        }

        if (useMulti && multiPurpose === "distributed") {
          return { sid, plan, compare: null };
        }

        if (allowWarmup) {
          await runWarmupOnServer({ sid, plan, label: src.label || sid, remoteServers: [] });
        }
        return { sid, plan, compare };
      }

      const results = [];
      if (useMulti && multiPurpose === "distributed") {
        const labels = targets.map((sid) => (sources.find((s) => s.id === sid) || {}).label || sid);
        log(`开始静态资源分析：${labels.join(" + ")}`);
        log("说明：静态计划为各服务器 StrategyPlan 合并；若开启 Warmup，将使用真实分布式 RPC 进行算法评估。");
      }
      for (const sid of targets) {
        const item = await analyzeOne(sid);
        results.push(item);
      }

      if (results.length <= 1) {
        if (results.length === 1 && results[0].plan) {
          lastEvalPlan = results[0].plan;
          lastEvalMode = "single";
          lastEvalTarget = results[0].sid;
          setAnalyzeLockReady(true);
        }
        return;
      }

      if (useMulti && multiPurpose === "distributed") {
        try {
          const resp = await fetch("/api/distributed_strategy_plan", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ servers: targets }),
          });
          const data = await resp.json();
          if (!resp.ok) {
            log(`DistributedStrategyPlan 失败：${JSON.stringify(data)}`);
            return;
          }
          lastEvalPlan = data;
          lastEvalMode = "distributed";
          lastEvalTarget = null;
          setAnalyzeLockReady(true);

          const cards = targets
            .map((sid) => {
              const src = sources.find((s) => s.id === sid) || {};
              const snap = (data.server_resources || {})[sid] || {};
              const cpu = snap.cpu || {};
              const mem = snap.memory || {};
              const gpus = Array.isArray(snap.gpus) ? snap.gpus : [];
              return `
                <div style="display:grid;gap:6px;padding:10px;border-radius:10px;background:#f8fafc;border:1px solid #e2e8f0;">
                  <div style="font-weight:900;">${src.label || sid}</div>
                  <div style="font-size:12px;color:var(--muted);">CPU ${pct(cpu.usage_percent)} · MEM ${pct(mem.percent)} · GPU ${gpus.length} 张</div>
                </div>
              `;
            })
            .join("");

          const selection = targets
            .map((sid) => {
              const src = sources.find((s) => s.id === sid) || {};
              const info = (data.servers || {})[sid] || {};
              if (info.backend === "cpu") return `${src.label || sid}: CPU`;
              const devs = Array.isArray(info.devices) && info.devices.length ? info.devices.join(",") : "-";
              return `${src.label || sid}: GPU ${devs}`;
            })
            .join(" · ");

          gaPanel.innerHTML = `
            <div style="display:grid;gap:10px;">
              <div style="font-weight:900;">分布式资源汇总（多机多卡）</div>
              ${cards}
              <div style="font-size:12px;color:var(--muted);">最佳多GPU选择：${selection}</div>
            </div>
          `;
          if (allowWarmup && targets.length) {
            const primary = targets[0];
            const label = (sources.find((s) => s.id === primary) || {}).label || primary;
            const plan = (data.servers || {})[primary] || null;
            const remote = targets.filter((t) => t !== primary);
            await runWarmupOnServer({ sid: primary, plan, label, remoteServers: remote });
          }
        } catch (e) {
          log(`DistributedStrategyPlan 异常：${e}`);
        }
        return;
      }

      const bestRows = results
        .map(({ sid, plan }) => {
          if (!plan || plan.estimated_time_ms == null) return null;
          const label = plan.backend === "cpu"
            ? "CPU"
            : plan.backend === "cuda"
              ? `GPU(${(plan.devices || [])[0] ?? "-"})`
              : plan.backend === "multi-gpu"
                ? `Multi-GPU(${(plan.devices || []).length})`
                : (plan.backend || "best");
          return { sid, label, ms: plan.estimated_time_ms };
        })
        .filter(Boolean);

      if (!bestRows.length) return;

      const maxMs = Math.max(...bestRows.map((r) => r.ms || 0), 1);
      const cards = bestRows
        .map((r) => {
          const src = sources.find((s) => s.id === r.sid) || {};
          const pct = Math.max(2, Math.round((r.ms / maxMs) * 100));
          return `
            <div style="display:grid;gap:6px;padding:10px;border-radius:10px;background:#f8fafc;border:1px solid #e2e8f0;">
              <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
                <div style="font-weight:900;">${src.label || r.sid}</div>
                <div style="font-size:12px;color:var(--muted);white-space:nowrap;">${r.label} · ${r.ms.toFixed(2)} ms</div>
              </div>
              <div style="height:10px;background:#e2e8f0;border-radius:999px;overflow:hidden;">
                <div style="width:${pct}%;height:100%;background:#2563eb;"></div>
              </div>
            </div>
          `;
        })
        .join("");

      gaPanel.innerHTML = `
        <div style="display:grid;gap:10px;">
          <div style="font-weight:900;">多机最佳方案对比</div>
          ${cards}
        </div>
      `;
      } catch (e) {
        log(`评估启动失败：${e}`);
        console.error("eval start failed", e);
      } finally {}
    });

    // ---------- Module 3: run & monitor ----------
    const HISTORY_KEY = "gapa_history_v1";
    let historyCache = [];
    let historySelected = new Set();

    async function historyImport(items) {
      if (!Array.isArray(items) || !items.length) return false;
      try {
        const resp = await fetch("/api/history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ items }),
        });
        if (!resp.ok) return false;
        const data = await resp.json();
        historyCache = Array.isArray(data.items) ? data.items : historyCache;
        return true;
      } catch (e) {
        return false;
      }
    }

    async function historyLoad() {
      try {
        const resp = await fetch("/api/history", { cache: "no-store" });
        if (!resp.ok) return [];
        const items = await resp.json();
        historyCache = Array.isArray(items) ? items : [];
        if (!historyCache.length) {
          try {
            const legacy = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]") || [];
            if (legacy.length) {
              const ok = await historyImport(legacy);
              if (ok) {
                localStorage.removeItem(HISTORY_KEY);
                return historyCache;
              }
            }
          } catch (e) {}
        }
        return historyCache;
      } catch (e) {
        return [];
      }
    }
    async function historyAdd(item) {
      try {
        const resp = await fetch("/api/history", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(item),
        });
        if (!resp.ok) return false;
        const data = await resp.json();
        historyCache = Array.isArray(data.items) ? data.items : historyCache;
        return true;
      } catch (e) {
        return false;
      }
    }
    async function historyDelete(ids) {
      if (!Array.isArray(ids) || !ids.length) return false;
      try {
        const resp = await fetch("/api/history", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ ids }),
        });
        if (!resp.ok) return false;
        const data = await resp.json();
        historyCache = Array.isArray(data.items) ? data.items : historyCache;
        return true;
      } catch (e) {
        return false;
      }
    }
    async function historyClear() {
      try {
        const resp = await fetch("/api/history", {
          method: "DELETE",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ all: true }),
        });
        if (!resp.ok) return false;
        historyCache = [];
        return true;
      } catch (e) {
        return false;
      }
    }

    function updateHistoryActions() {
      const hasSelection = historySelected.size > 0;
      if (btnHistoryDelete) btnHistoryDelete.disabled = !hasSelection;
      if (historySelectAll) {
        historySelectAll.checked = historyCache.length > 0 && historySelected.size === historyCache.length;
        historySelectAll.indeterminate = historySelected.size > 0 && historySelected.size < historyCache.length;
      }
    }

    function historyRender() {
      const items = historyCache;
      if (!items.length) {
        historyList.innerHTML = `<div style="color:var(--muted);font-size:13px;">暂无历史记录。</div>`;
        historySelected = new Set();
        updateHistoryActions();
        return;
      }
      historyList.innerHTML = items
        .map((it, idx) => {
          const commMs = it.comm_avg_ms != null ? `${Number(it.comm_avg_ms).toFixed(2)}ms` : "-";
          const checked = historySelected.has(it.id) ? "checked" : "";
          return `
            <div class="history-item" data-idx="${idx}" data-id="${it.id || ""}">
              <div class="history-item-row">
                <input type="checkbox" class="history-item-check" data-id="${it.id || ""}" ${checked} />
                <div>
                  <div style="font-weight:900;">${it.algorithm} · ${it.server_label || it.server_id}</div>
                  <div class="history-meta">${it.timestamp} · state=${it.state} · best=${it.best_score ?? "-"} · comm=${commMs}</div>
                </div>
              </div>
            </div>
          `;
        })
        .join("");
      historyList.querySelectorAll(".history-item-check").forEach((el) => {
        el.addEventListener("click", (evt) => {
          evt.stopPropagation();
          const id = el.dataset.id;
          if (!id) return;
          if (el.checked) historySelected.add(id);
          else historySelected.delete(id);
          updateHistoryActions();
        });
      });
      historyList.querySelectorAll(".history-item").forEach((el) => {
        el.addEventListener("click", (evt) => {
          if (evt.target && evt.target.classList.contains("history-item-check")) return;
          historyList.querySelectorAll(".history-item").forEach((x) => x.classList.remove("history-item--active"));
          el.classList.add("history-item--active");
          const idx = Number(el.dataset.idx);
          const it = historyCache[idx];
          if (!it) return;
          const comm = it.result?.comm || {};
          const perRank = comm.per_rank_avg_ms || {};
          const perOps = comm.per_rank_ops || {};
          const perMeta = comm.per_rank_meta || {};
          const commLines = Object.keys(perRank)
            .filter((k) => k !== "0")
            .map((k) => {
              const avg = Number(perRank[k]).toFixed(2);
              const meta = perMeta[k] || {};
              const gpuNameRaw = meta.gpu_name_short || meta.gpu_name || "";
              const gpuName = gpuNameRaw ? ` ${gpuNameRaw}` : "";
              const label = meta.host ? `${meta.host}:gpu${meta.gpu ?? k}${gpuName}` : `rank${k}${gpuName}`;
              const ops = perOps[k] || {};
              const topOps = Object.keys(ops)
                .sort((a, b) => (ops[b] || 0) - (ops[a] || 0))
                .slice(0, 3)
                .map((op) => `${op} ${Number(ops[op]).toFixed(1)}ms`);
              const opText = topOps.length ? ` (${topOps.join(", ")})` : "";
              return `comm_rank: ${label} ${avg}ms${opText}`;
            });
          historyDetail.textContent = [
            `# ${it.algorithm} @ ${it.server_label || it.server_id}`,
            `time: ${it.timestamp}`,
            `state: ${it.state}`,
            `task_id: ${it.task_id}`,
            it.dataset ? `dataset: ${it.dataset}` : "",
            it.hyperparams ? `hyperparams: ${JSON.stringify(it.hyperparams)}` : "",
            it.best_score != null ? `best_score: ${it.best_score}` : "",
            it.comm_avg_ms != null ? `comm_avg_ms: ${Number(it.comm_avg_ms).toFixed(2)}` : "",
            ...commLines,
            "",
            ...(it.logs || []),
          ]
            .filter(Boolean)
            .join("\n");
        });
      });
      updateHistoryActions();
    }
    btnHistoryClear?.addEventListener("click", async () => {
      const ok = await historyClear();
      if (ok) {
        historyRender();
        historyDetail.textContent = "历史已清空。";
      }
    });
    btnHistoryDelete?.addEventListener("click", async () => {
      const ids = Array.from(historySelected);
      if (!ids.length) return;
      const ok = await historyDelete(ids);
      if (ok) {
        historySelected = new Set();
        historyRender();
        historyDetail.textContent = "已删除选中历史。";
      }
    });
    historySelectAll?.addEventListener("change", () => {
      if (!historyCache.length) return;
      historySelected = new Set();
      if (historySelectAll.checked) {
        historyCache.forEach((it) => it.id && historySelected.add(it.id));
      }
      historyRender();
    });

    function appendRunLog(lines) {
      if (!lines || !lines.length) return;
      if (runLog.textContent.trim() === "等待任务…") runLog.textContent = "";
      runLog.textContent += lines.join("\n") + "\n";
      runLog.scrollTop = runLog.scrollHeight;
    }
    btnRunClear?.addEventListener("click", () => {
      runLog.textContent = "";
      lastLogCount = 0;
    });

    function renderRunCharts(result) {
      if (!result) return;
      const curves = result.curves && typeof result.curves === "object" ? result.curves : null;
      const objectives = result.objectives && typeof result.objectives === "object" ? result.objectives : null;
      const primaryName = objectives?.primary || null;
      const secondaryName = objectives?.secondary || null;
      const primaryArr = primaryName && curves && Array.isArray(curves[primaryName]) ? curves[primaryName] : null;
      const secondaryArr = secondaryName && curves && Array.isArray(curves[secondaryName]) ? curves[secondaryName] : null;
      const conv = primaryArr || (Array.isArray(result.convergence) ? result.convergence : []);
      const metrics = Array.isArray(result.metrics) ? result.metrics : [];
      if (!conv.length && !metrics.length) return;

      // convergence polyline
      const w = 520, h = 140, pad = 18;
      function polylineFor(arr) {
        if (!arr || arr.length < 2) return "";
        const minV = Math.min(...arr);
        const maxV = Math.max(...arr);
        return arr
          .map((y, i) => {
            const x = pad + (i / (arr.length - 1)) * (w - pad * 2);
            const yy = pad + (1 - (y - minV) / Math.max(1e-9, maxV - minV)) * (h - pad * 2);
            return `${x.toFixed(1)},${yy.toFixed(1)}`;
          })
          .join(" ");
      }

      const cpuArr = metrics.map((m) => m.cpu_usage_percent).filter((x) => x != null);
      const memArr = metrics.map((m) => m.memory_percent).filter((x) => x != null);
      const cpuPts = polylineFor(cpuArr);
      const memPts = polylineFor(memArr);
      const convPts = polylineFor(conv);
      const conv2Pts = secondaryArr ? polylineFor(secondaryArr) : "";

      const title = primaryName
        ? `指标曲线：${primaryName}${secondaryName ? " / " + secondaryName : ""}`
        : "指标曲线";
      runChart.innerHTML = `
        <div style="width:100%;display:grid;gap:12px;">
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
            <div style="font-weight:900;">${title}</div>
            <div style="font-size:12px;color:var(--muted);">
              <span style="display:inline-block;width:10px;height:10px;background:#2563eb;border-radius:3px;margin-right:6px;"></span>${primaryName || "primary"}
              ${secondaryName ? `<span style="display:inline-block;width:10px;height:10px;background:#a855f7;border-radius:3px;margin:0 6px 0 12px;"></span>${secondaryName}` : ""}
            </div>
          </div>
          <div style="border:1px dashed var(--card-border);border-radius:10px;background:var(--soft-bg);padding:10px;overflow:hidden;">
            <svg viewBox="0 0 ${w} ${h}" width="100%" height="140" preserveAspectRatio="none">
              ${convPts ? `<polyline fill="none" stroke="#2563eb" stroke-width="3" points="${convPts}" />` : ""}
              ${conv2Pts ? `<polyline fill="none" stroke="#a855f7" stroke-width="3" points="${conv2Pts}" />` : ""}
              <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#cbd5e1" stroke-width="2"/>
              <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="#cbd5e1" stroke-width="2"/>
            </svg>
          </div>
          <div style="display:flex;align-items:center;justify-content:space-between;gap:10px;">
            <div style="font-weight:900;">轮次资源监视（每轮采样）</div>
            <div style="font-size:12px;color:var(--muted);">
              <span style="display:inline-block;width:10px;height:10px;background:#f59e0b;border-radius:3px;margin-right:6px;"></span>CPU%
              <span style="display:inline-block;width:10px;height:10px;background:#16a34a;border-radius:3px;margin:0 6px 0 12px;"></span>MEM%
            </div>
          </div>
          <div style="border:1px dashed var(--card-border);border-radius:10px;background:var(--soft-bg);padding:10px;overflow:hidden;">
            <svg viewBox="0 0 ${w} ${h}" width="100%" height="140" preserveAspectRatio="none">
              ${cpuPts ? `<polyline fill="none" stroke="#f59e0b" stroke-width="3" points="${cpuPts}" />` : ""}
              ${memPts ? `<polyline fill="none" stroke="#16a34a" stroke-width="3" points="${memPts}" />` : ""}
              <line x1="${pad}" y1="${h - pad}" x2="${w - pad}" y2="${h - pad}" stroke="#cbd5e1" stroke-width="2"/>
              <line x1="${pad}" y1="${pad}" x2="${pad}" y2="${h - pad}" stroke="#cbd5e1" stroke-width="2"/>
            </svg>
          </div>
        </div>
      `;
    }

    async function startRun() {
      const selectedServer = runServer?.value || RUN_SERVER_ID;
      const server_id = selectedServer === "all" ? "local" : selectedServer;
      const algorithm = runAlgo.value;
      const dataset = runDataset?.value || "";
      const iterations = Math.max(1, Number(runIters.value || 1));
      const crossover_rate = Math.min(1, Math.max(0, Number(runPc.value || 0.8)));
      const mutate_rate = Math.min(1, Math.max(0, Number(runPm.value || 0.2)));
      let mode = (runMode.value || "AUTO").toUpperCase();

      let devices = [];
      const snap = snapshots.get(server_id) || {};
      const hasGpu = Array.isArray(snap.gpus) && snap.gpus.length > 0;
      let remote_servers = [];

      if (mode === "AUTO" || mode === "MNM") {
        try {
          const resp = await fetch(`/api/resource_lock/status?scope=all`, { cache: "no-store" });
          const data = await resp.json();
          if (resp.ok) {
            const results = data.results || {};
            remote_servers = Object.keys(results).filter((k) => k !== "local" && results[k]?.active);
            if (selectedServer === "all") {
              mode = "MNM";
            }
          }
        } catch (e) {
          remote_servers = [];
        }
      }

      if (mode === "AUTO") {
        const saved = getSavedPlan(server_id);
        if (saved && saved.plan && Array.isArray(saved.plan.devices)) {
          devices = saved.plan.devices;
        }
      } else if (mode === "CPU") {
        devices = [];
      } else if ((mode === "S" || mode === "SM") && hasGpu) {
        devices = [Number(runGpuSingle.value)];
      } else if ((mode === "M" || mode === "MNM") && hasGpu) {
        const checked = Array.from(runGpuMulti.querySelectorAll(".run-gpu-check"))
          .filter((el) => el.checked)
          .map((el) => Number(el.value));
        devices = checked;
      }
      if (mode === "M" && (!hasGpu || !devices.length)) {
        appendRunLog([`[WARN] 未选中多卡，已切换为 S。`]);
        mode = "S";
        if (hasGpu) {
          devices = [Number(runGpuSingle.value)];
        } else {
          devices = [];
        }
      }

      if (mode !== "MNM") {
        remote_servers = [];
      }

      // reset UI
      runState.textContent = "starting";
      runTaskId.textContent = "-";
      runProgress.style.width = "0%";
      runProgressText.textContent = "0%";
      runBest.textContent = "-";
      if (runComm) runComm.textContent = "-";
      if (runCommDetail) runCommDetail.textContent = "";
      runChart.textContent = "等待数据...";
      runLog.textContent = "";
      lastLogCount = 0;

      const resp = await fetch("/api/analysis/start", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          server_id,
          algorithm,
          dataset,
          iterations,
          crossover_rate,
          mutate_rate,
          mode,
          devices,
          remote_servers,
          release_lock_on_finish: true,
          timeout_s: 20,
        }),
      });
      if (!resp.ok) {
        let errBody = null;
        try { errBody = await resp.json(); } catch (e) { errBody = { raw: await resp.text() }; }
        appendRunLog([`[ERROR] start failed: HTTP ${resp.status} ${JSON.stringify(errBody)}`]);
        runState.textContent = "error";
        return;
      }
      const data = await resp.json();
      runTaskId.textContent = data.task_id || "-";
      runState.textContent = data.status || "started";
      appendRunLog([`[INFO] task started: ${data.task_id}`]);

      // poll status
      if (currentPoll) clearInterval(currentPoll);
      currentPoll = setInterval(() => pollRunStatus(server_id), 1200);
      pollRunStatus(server_id);
    }

    async function stopRun() {
      const server_id = RUN_SERVER_ID;
      try {
        btnRunStop.disabled = true;
        const resp = await fetch(`/api/analysis/stop`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ server_id }),
        });
        if (!resp.ok) {
          appendRunLog([`[ERROR] stop failed: HTTP ${resp.status}`]);
        } else {
          const data = await resp.json();
          appendRunLog([`[WARN] stop requested: ${JSON.stringify(data)}`]);
        }
      } catch (e) {
        appendRunLog([`[ERROR] stop failed: ${e}`]);
      }
    }

    async function pollRunStatus(server_id) {
      const resp = await fetch(`/api/analysis/status?server_id=${encodeURIComponent(server_id)}&timeout_s=10`, { cache: "no-store" });
      if (!resp.ok) return;
      const st = await resp.json();
      runState.textContent = st.state || "unknown";
      btnRunStop.disabled = !(st.state === "running");
      const p = Number(st.progress || 0);
      runProgress.style.width = `${Math.max(0, Math.min(100, p))}%`;
      runProgressText.textContent = `${Math.max(0, Math.min(100, p))}%`;

      const logs = Array.isArray(st.logs) ? st.logs : [];
      if (logs.length > lastLogCount) {
        appendRunLog(logs.slice(lastLogCount));
        lastLogCount = logs.length;
      }

      if (st.result) {
        const best = st.result.best_score;
        if (best != null) runBest.textContent = String(best);
        if (runComm) {
          const comm = st.result.comm || {};
          const avgMs = comm.avg_ms;
          runComm.textContent = avgMs ? `${Number(avgMs).toFixed(2)} ms` : "-";
          if (runCommDetail) {
            const perRank = comm.per_rank_avg_ms || {};
            const perOps = comm.per_rank_ops || {};
            const perMeta = comm.per_rank_meta || {};
            const items = Object.keys(perRank)
              .filter((k) => k !== "0")
              .map((k) => {
                const avg = Number(perRank[k]).toFixed(2);
                const meta = perMeta[k] || {};
                const gpuNameRaw = meta.gpu_name_short || meta.gpu_name || "";
                const gpuName = gpuNameRaw ? ` ${gpuNameRaw}` : "";
                const label = meta.host ? `${meta.host}:gpu${meta.gpu ?? k}${gpuName}` : `rank${k}${gpuName}`;
                const ops = perOps[k] || {};
                const topOps = Object.keys(ops)
                  .sort((a, b) => (ops[b] || 0) - (ops[a] || 0))
                  .slice(0, 2)
                  .map((op) => `${op} ${Number(ops[op]).toFixed(1)}ms`);
                const opText = topOps.length ? ` (${topOps.join(", ")})` : "";
                return `${label} ${avg}ms${opText}`;
              });
            runCommDetail.textContent = items.length ? items.join(" / ") : "";
          }
        }
        renderRunCharts(st.result);
      }

      if (st.state === "completed" || st.state === "error" || st.state === "idle") {
        if (currentPoll) {
          clearInterval(currentPoll);
          currentPoll = null;
        }
        // persist to history
        const saved = {
          timestamp: new Date().toLocaleString(),
          server_id,
          server_label: (sources.find((s) => s.id === server_id) || {}).label,
          algorithm: runAlgo.value,
          dataset: runDataset?.value || null,
          task_id: st.task_id,
          state: st.state,
          best_score: st.result ? st.result.best_score : null,
          comm_avg_ms: st.result && st.result.comm && st.result.comm.avg_ms != null ? st.result.comm.avg_ms : null,
          hyperparams: st.result ? st.result.hyperparams : null,
          logs: logs.slice(-500),
          result: st.result || null,
        };
        const ok = await historyAdd(saved);
        if (ok) {
          historyRender();
        }
      }
    }

    btnRun?.addEventListener("click", startRun);
    btnRunStop?.addEventListener("click", stopRun);

    async function refreshRunLockStatus() {
      try {
        const resp = await fetch(`/api/resource_lock/status?scope=all`, { cache: "no-store" });
        const data = await resp.json();
        if (!resp.ok) {
          if (runLockStatus) runLockStatus.textContent = "锁定状态获取失败";
          return;
        }
        const results = data.results || {};
        updateRunServerOptions(results);
        const sel = runServer?.value || RUN_SERVER_ID;
        const activeRemote = Object.keys(results).filter((k) => k !== "local" && results[k]?.active);
        if (sel === "all") {
          if (activeRemote.length) {
            runLockStatus.textContent = "分布式锁定已启用";
            runLockDetail.textContent = `远程服务器：${activeRemote.join(", ")}`;
          } else {
            runLockStatus.textContent = "未锁定";
            runLockDetail.textContent = "未锁定时将默认使用 CPU 运行";
          }
          return;
        }
        const info = results[sel] || {};
        if (info?.active) {
          const label = (sources.find((s) => s.id === sel) || {}).label || sel;
          runLockStatus.textContent = `${label} 已锁定`;
          const devices = Array.isArray(info.devices) ? info.devices.join(",") : "-";
          const exp = info.expires_at ? new Date(info.expires_at * 1000).toLocaleString() : "-";
          runLockDetail.textContent = `backend=${info.backend} devices=${devices} expires=${exp}`;
          const devs = Array.isArray(info.devices) ? info.devices.map((d) => Number(d)) : [];
          if (info.backend === "cuda" && runMode) {
            runMode.value = "S";
          } else if (info.backend === "multi-gpu" && runMode) {
            runMode.value = "M";
          }
          updateRunModeVisibility();
          updateRunGpuOptions();
          if (devs.length) {
            if (runGpuSingle) runGpuSingle.value = String(devs[0]);
            if (runGpuMulti) {
              runGpuMulti.querySelectorAll(".run-gpu-check").forEach((el) => {
                el.checked = devs.includes(Number(el.value));
              });
            }
          }
        } else {
          runLockStatus.textContent = "未锁定";
          runLockDetail.textContent = "未锁定时将默认使用 CPU 运行";
        }
      } catch (e) {
        if (runLockStatus) runLockStatus.textContent = "锁定状态获取失败";
      }
    }

    btnRunLockRefresh?.addEventListener("click", refreshRunLockStatus);
    btnRunLockRelease?.addEventListener("click", () => releaseLock(RUN_SERVER_ID));

    async function applyEvalPlanToLock() {
      if (!lastEvalPlan || !lastEvalMode) return;
      try {
        const stResp = await fetch(`/api/resource_lock/status?scope=all`, { cache: "no-store" });
        const stData = await stResp.json();
        const results = stData.results || {};
        const hasActive = Object.values(results).some((v) => v && v.active);
        if (hasActive) {
          alert("已有资源锁定在运行，请先释放锁定。");
          return;
        }
      } catch (e) {
        // ignore status errors, continue to attempt lock
      }

      if (lastEvalMode === "distributed") {
        const devicesByServer = lastEvalPlan.devices_by_server || {};
        if (!Object.keys(devicesByServer).length) {
          alert("分布式评估未返回可锁定的设备。");
          return;
        }
        const resp = await fetch("/api/resource_lock", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scope: "all", devices_by_server: devicesByServer, strict_idle: true }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          alert(`资源锁定失败: ${JSON.stringify(data)}`);
          return;
        }
        lockLog(`已应用分布式锁定策略: ${JSON.stringify(data.results || data)}`);
        fetchLockStatus("all");
        closeEvalModal();
        return;
      }

      const devices = Array.isArray(lastEvalPlan.devices) ? lastEvalPlan.devices : [];
      if (!devices.length) {
        alert("评估结果未包含可用设备，无法锁定。");
        return;
      }
      const scope = lastEvalTarget || "local";
      const resp = await fetch("/api/resource_lock", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ scope, devices }),
      });
      const data = await resp.json();
      if (!resp.ok) {
        alert(`资源锁定失败: ${JSON.stringify(data)}`);
        return;
      }
      lockLog(`已应用锁定策略: ${JSON.stringify(data.results || data)}`);
      fetchLockStatus("local");
      closeEvalModal();
    }

    btnAnalyzeLock?.addEventListener("click", applyEvalPlanToLock);

    async function applyLock() {
      const scope = lockScope?.value || "all";
      const mode = (lockMode?.value || "auto").toLowerCase();
      const duration_s = Math.max(10, Number(lockDuration.value || 600));
      const warmup_iters = Math.max(0, Number(lockWarmup.value || 2));
      const mem_mb = Math.max(0, Number(lockMem.value || 0));
      let devices = null;
      let devices_by_server = null;
      if (mode === "manual") {
        const checked = Array.from(lockServerList.querySelectorAll(".lock-gpu-check"))
          .filter((el) => el.checked)
          .map((el) => ({ server: el.dataset.server, id: Number(el.value) }));
        if (!checked.length) {
          alert("请选择至少一张显卡");
          return;
        }
        devices_by_server = {};
        checked.forEach((item) => {
          if (!devices_by_server[item.server]) devices_by_server[item.server] = [];
          devices_by_server[item.server].push(item.id);
        });
        if (scope !== "all" && devices_by_server[scope]) {
          devices = devices_by_server[scope];
        }
      }
      try {
        lockLog(`开始锁定资源: scope=${scope}, mode=${mode}, mem=${mem_mb}MB`);
        const resp = await fetch("/api/resource_lock", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scope, duration_s, warmup_iters, mem_mb, devices, devices_by_server }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          lockLog(`锁定失败: ${JSON.stringify(data)}`);
          alert(`资源锁定失败: ${JSON.stringify(data)}`);
          return;
        }
        lockLog(`锁定完成: ${JSON.stringify(data.results || data)}`);
        fetchLockStatus(scope);
        closeLockModal();
      } catch (e) {
        lockLog(`锁定失败: ${e}`);
        alert(`资源锁定失败: ${e}`);
      }
    }

    async function releaseLock(scopeOverride) {
      const scope = scopeOverride || lockScope?.value || "all";
      try {
        const resp = await fetch("/api/resource_lock/release", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ scope }),
        });
        const data = await resp.json();
        if (!resp.ok) {
          lockLog(`释放失败: ${JSON.stringify(data)}`);
          alert(`释放失败: ${JSON.stringify(data)}`);
          return;
        }
        lockLog(`已释放资源锁定: ${JSON.stringify(data.results || data)}`);
        fetchLockStatus(scope);
        closeLockModal();
      } catch (e) {
        lockLog(`释放失败: ${e}`);
        alert(`释放失败: ${e}`);
      }
    }

    lockApply?.addEventListener("click", applyLock);
    lockRelease?.addEventListener("click", releaseLock);
    btnLockStatus?.addEventListener("click", () => fetchLockStatus(lockStatusScope?.value || "all"));
    btnLockReleaseNow?.addEventListener("click", () => {
      const scope = lockStatusScope?.value || "all";
      releaseLock(scope);
      fetchLockStatus(scope);
    });

    // Boot
    (async () => {
      await loadAlgorithms();
      await loadDatasets();
      await loadConfigServers();
      updateLockScopeOptions();
      updateServerMultiList();
      refreshAll();
      fetchLockStatus("all");
      await refreshRunLockStatus();
      applyRunConfig();
      await historyLoad();
      historyRender();
      updateRunDatasetOptions();
      updateEvalDatasetOptions();
      if (evalDataset) evalDataset.disabled = algoSelect?.value === "monitor";
    })();
