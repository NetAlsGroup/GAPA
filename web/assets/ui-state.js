const state = {
  run: {
    taskId: "",
    status: "idle",
    modeDecision: null,
    degradeReason: "",
    queueSize: 0,
  },
  lock: {
    scope: "local",
    active: false,
  },
};

export function setRunState(next = {}) {
  state.run = { ...state.run, ...next };
  return state.run;
}

export function setModeDecision(modeDecision) {
  const md = modeDecision || null;
  state.run.modeDecision = md;
  state.run.degradeReason = md?.reason || "";
  return state.run.modeDecision;
}

export function snapshotUiState() {
  return {
    run: { ...state.run },
    lock: { ...state.lock },
  };
}
