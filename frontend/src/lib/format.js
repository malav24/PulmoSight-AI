export const fmt = (v, digits = 3) => {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  if (typeof v === "number") return v.toFixed(digits);
  return String(v);
};

export const pct = (v, digits = 1) =>
  v === null || v === undefined || Number.isNaN(v) ? "—" : `${(v * 100).toFixed(digits)}%`;

export const uncertaintyLevel = (std) => {
  if (std === null || std === undefined) return { label: "Not computed", tone: "neutral" };
  if (std < 0.05) return { label: "Low", tone: "good" };
  if (std < 0.15) return { label: "Moderate", tone: "warn" };
  return { label: "High", tone: "bad" };
};

export const MODEL_ORDER = ["model_b", "model_a", "efficientnet_b0", "densenet121", "resnet50", "vit_small", "ablation_cls_only"];

export const SERIES_COLORS = {
  model_b: "#0e8f86",
  model_a: "#2f6fd6",
  efficientnet_b0: "#8b5cf6",
  densenet121: "#d97706",
  resnet50: "#64748b",
  vit_small: "#db2777",
  ablation_cls_only: "#94a3b8",
};

export const MAP_INFO = {
  localization: {
    title: "Localization heatmap",
    plain: "Where the model predicts the lung opacity is.",
    gradient: "linear-gradient(90deg,#440154,#3b528b,#21918c,#5ec962,#fde725)",
    scale: "per-pixel probability 0 → 1",
  },
  gradcam: {
    title: "Grad-CAM++",
    plain: "Which regions pushed the model's score up (an after-the-fact explanation).",
    gradient: "linear-gradient(90deg,#000004,#420a68,#932667,#dd513a,#fca50a,#fcffa4)",
    scale: "relative influence, low → high",
  },
  attention: {
    title: "Spatial attention",
    plain: "How the hybrid classifier weighted image regions internally.",
    gradient: "linear-gradient(90deg,#00224e,#3a4c6e,#7d7c78,#bcaf6f,#fee838)",
    scale: "relative weight, low → high",
  },
};
