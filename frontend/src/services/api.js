// Centralised API client. Every backend URL lives here - components never build URLs themselves.

const BASE = (import.meta.env.VITE_API_BASE_URL || "/api").replace(/\/$/, "");

export class ApiError extends Error {
  constructor(status, code, message, details) {
    super(message);
    this.status = status;
    this.code = code;
    this.details = details;
  }
}

const NETWORK_MESSAGE =
  "Cannot reach the analysis server. Check that the backend is running (uvicorn backend.app.main:app --port 8000).";

async function parseError(res) {
  let body = null;
  try {
    body = await res.json();
  } catch {
    /* non-JSON error */
  }
  const e = body?.error;
  return new ApiError(res.status, e?.code || "http_error", e?.message || `Request failed (${res.status})`, e?.details);
}

async function request(method, path, { json, signal } = {}) {
  let res;
  try {
    res = await fetch(`${BASE}${path}`, {
      method,
      signal,
      headers: json ? { "Content-Type": "application/json" } : undefined,
      body: json ? JSON.stringify(json) : undefined,
    });
  } catch (err) {
    if (err.name === "AbortError") throw err;
    throw new ApiError(0, "network_error", NETWORK_MESSAGE);
  }
  if (!res.ok) throw await parseError(res);
  return res.status === 204 ? null : res.json();
}

// Multipart upload with progress (fetch has no upload progress events).
function upload(path, fields, onProgress) {
  return new Promise((resolve, reject) => {
    const form = new FormData();
    Object.entries(fields).forEach(([k, v]) => v !== undefined && v !== null && v !== "" && form.append(k, v));
    const xhr = new XMLHttpRequest();
    xhr.open("POST", `${BASE}${path}`);
    xhr.responseType = "json";
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.({ phase: "upload", fraction: e.loaded / e.total });
    xhr.upload.onload = () => onProgress?.({ phase: "processing", fraction: 1 });
    xhr.onerror = () => reject(new ApiError(0, "network_error", NETWORK_MESSAGE));
    xhr.onload = () => {
      const body = xhr.response;
      if (xhr.status >= 200 && xhr.status < 300) return resolve(body);
      const e = body?.error;
      reject(new ApiError(xhr.status, e?.code || "http_error", e?.message || `Request failed (${xhr.status})`, e?.details));
    };
    xhr.send(form);
  });
}

export const api = {
  health: (o) => request("GET", "/health", o),
  models: (o) => request("GET", "/models", o),
  loadModel: (modelKey, reload = false) => request("POST", "/models/load", { json: { model_key: modelKey, reload } }),
  datasets: (refresh = false) => request("GET", `/datasets/status${refresh ? "?refresh=true" : ""}`),
  analyze: (file, modelKey, onProgress) => upload("/analyze", { file, model_key: modelKey }, onProgress),
  explain: (file, modelKey, referenceBoxes, onProgress) =>
    upload("/explain", { file, model_key: modelKey, reference_boxes: referenceBoxes }, onProgress),
  analyses: (limit = 10) => request("GET", `/analyses?limit=${limit}`),
  experimentResults: () => request("GET", "/experiments/results"),
  runExperiment: (kind, modelKey, params = {}) =>
    request("POST", "/experiments/run", { json: { kind, model_key: modelKey || null, params } }),
  job: (jobId) => request("GET", `/experiments/jobs/${jobId}`),
};

export const urls = {
  analysisReport: (analysisId) => `${BASE}/research/report?analysis_id=${encodeURIComponent(analysisId)}`,
  researchReport: () => `${BASE}/research/report`,
  resultFile: (name) => `${BASE}/experiments/files/${name.split("/").map(encodeURIComponent).join("/")}`,
  apiDocs: () => `${BASE.replace(/\/api$/, "")}/docs`,
};

export const UPLOAD_LIMIT_MB = 25;
export const ACCEPTED = [".png", ".jpg", ".jpeg", ".dcm", ".dicom"];
