const DEFAULTS = {
  retries: 2,
  timeoutMs: 10000,
  retryStatus: [408, 429, 500, 502, 503, 504],
};

function isApiUrl(input) {
  const raw = typeof input === "string" ? input : String(input?.url || "");
  return raw.startsWith("/api/") || raw.includes("/api/");
}

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export function installApiClient(opts = {}) {
  if (window.__gapaFetchInstalled) return window.fetch;
  const conf = {
    retries: Number(opts.retries ?? DEFAULTS.retries),
    timeoutMs: Number(opts.timeoutMs ?? DEFAULTS.timeoutMs),
    retryStatus: Array.isArray(opts.retryStatus) ? opts.retryStatus : DEFAULTS.retryStatus,
  };
  const nativeFetch = window.fetch.bind(window);

  window.fetch = async (input, init = {}) => {
    if (!isApiUrl(input)) {
      return nativeFetch(input, init);
    }

    const maxAttempts = Math.max(1, conf.retries + 1);
    let attempt = 0;
    let lastError = null;
    while (attempt < maxAttempts) {
      attempt += 1;
      const controller = new AbortController();
      const timeoutId = window.setTimeout(() => controller.abort(), conf.timeoutMs);
      try {
        const resp = await nativeFetch(input, { ...init, signal: controller.signal });
        window.clearTimeout(timeoutId);
        if (resp.ok || !conf.retryStatus.includes(resp.status) || attempt >= maxAttempts) {
          return resp;
        }
        await sleep(150 * Math.pow(2, attempt - 1));
        continue;
      } catch (err) {
        window.clearTimeout(timeoutId);
        lastError = err;
        if (attempt >= maxAttempts) {
          throw err;
        }
        await sleep(150 * Math.pow(2, attempt - 1));
      }
    }
    throw lastError || new Error("api fetch failed");
  };

  window.__gapaFetchInstalled = true;
  return window.fetch;
}
