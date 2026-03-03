export function formatModeDecision(md) {
  if (!md || typeof md !== "object") return "requested=- selected=- degraded=false reason=- code=-";
  return `requested=${md.requested_mode || "-"} selected=${md.selected_mode || "-"} degraded=${Boolean(md.degraded)} reason=${md.reason || "-"} code=${md.code || "-"}`;
}

export function renderModeDecisionLine(md) {
  return `[INFO] mode decision: ${formatModeDecision(md)}`;
}
