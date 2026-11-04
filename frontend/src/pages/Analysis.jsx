import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import {
  Activity, AlertCircle, CheckCircle2, Cpu, Crosshair, Eye, FileText, Gauge, HelpCircle, ScanLine, ShieldCheck,
} from "lucide-react";
import { useApp } from "../context/AppContext";
import { api, urls } from "../services/api";
import Dropzone from "../components/Dropzone";
import ImageViewer from "../components/ImageViewer";
import { NoTrainedModelNotice, ProbabilityMeter, ZoneBars } from "../components/common";
import {
  Badge, Button, Card, CardHeader, Collapsible, EmptyState, ErrorState, KeyValue, Notice, PageHeader, ProgressBar,
  Spinner, Tabs,
} from "../components/ui";
import { fmt, pct, uncertaintyLevel } from "../lib/format";

const OUTCOME_STYLE = {
  suspected: { tone: "warn", icon: AlertCircle },
  not_detected: { tone: "good", icon: CheckCircle2 },
  abstained: { tone: "neutral", icon: HelpCircle },
};

function OutcomeCard({ a }) {
  const p = a.prediction;
  const s = OUTCOME_STYLE[p.outcome];
  const Icon = s.icon;
  return (
    <Card>
      <CardHeader icon={Activity} title="Model output" subtitle="Pneumonia-related lung opacity pattern (RSNA definition)" />
      <div className="space-y-4 p-5">
        <div className={`flex items-start gap-3 rounded-lg p-4 ${p.outcome === "suspected" ? "bg-amber-50" : p.outcome === "not_detected" ? "bg-teal-soft" : "bg-slate-100"}`}>
          <Icon size={22} className={p.outcome === "suspected" ? "text-amber-700" : p.outcome === "not_detected" ? "text-teal-600" : "text-slate-500"} />
          <div>
            <p className="font-semibold text-navy">{p.label}</p>
            <p className="mt-0.5 text-xs text-navy-500">Model output only - not a diagnosis. A radiologist must interpret the image.</p>
          </div>
        </div>
        {a.configuration_warnings?.length > 0 && (
          <Notice tone="warn" title="Configuration warning">
            <ul className="list-disc pl-4">{a.configuration_warnings.map((w) => <li key={w}>{w}</li>)}</ul>
          </Notice>
        )}
        {p.abstained && (
          <Notice tone="warn" title="Why the system abstained">
            <ul className="list-disc pl-4">{p.abstention_reasons.map((r) => <li key={r}>{r}</li>)}</ul>
          </Notice>
        )}
        {p.calibrated_probability !== null && p.calibrated_probability !== undefined ? (
          <ProbabilityMeter label="Calibrated probability" value={p.calibrated_probability} />
        ) : (
          <p className="text-xs text-navy-500">Calibrated probability not available (calibration has not been fitted for this model).</p>
        )}
        <ProbabilityMeter label="Model probability (uncalibrated)" value={p.probability} threshold={p.threshold} />
        <Collapsible title="How to read these numbers">
          <p>The model gives a score between 0 and 1. Scores above the threshold ({fmt(p.threshold, 3)}, chosen on the validation set) are reported as “suspected”.
          The calibrated probability is adjusted with temperature scaling (T = {fmt(a.prediction.temperature, 2)}) so that, on the RSNA validation data, it better matches how often opacity was really annotated.
          Neither number is the probability that the patient has pneumonia.</p>
        </Collapsible>
      </div>
    </Card>
  );
}

function StatusCards({ a }) {
  const u = a.uncertainty || {};
  const lvl = uncertaintyLevel(u.std);
  const q = a.quality;
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <p className="flex items-center gap-2 text-sm font-semibold"><Gauge size={16} className="text-brand" /> Uncertainty</p>
          <Badge tone={lvl.tone}>{lvl.label}</Badge>
        </div>
        <p className="mt-2 text-xs text-navy-500">Spread of {u.n_samples ?? "—"} repeated predictions (MC dropout).</p>
        <KeyValue rows={[["Std. deviation", fmt(u.std)], ["5–95 % range", `${fmt(u.p05, 2)} – ${fmt(u.p95, 2)}`]]} />
      </Card>
      <Card className="p-4">
        <div className="flex items-center justify-between">
          <p className="flex items-center gap-2 text-sm font-semibold"><ShieldCheck size={16} className="text-brand" /> Input quality</p>
          <Badge tone={!q.passed ? "bad" : q.warnings.length ? "warn" : "good"}>{!q.passed ? "Failed" : q.warnings.length ? "Warnings" : "Passed"}</Badge>
        </div>
        <ul className="mt-2 space-y-1 text-xs text-navy-700">
          {[...q.errors, ...q.warnings].map((m) => <li key={m}>• {m}</li>)}
          {!q.errors.length && !q.warnings.length && <li>No problems found by the basic checks.</li>}
          <li className="text-navy-500">
            Distribution check: {!a.ood.available ? "not available (statistics not fitted)" : a.ood.flag ? "image looks unusual compared with the training data" : "within training range"}
          </li>
        </ul>
        <p className="mt-2 text-[11px] text-navy-300">{q.disclaimer}</p>
      </Card>
    </div>
  );
}

function ModelAndLocalization({ a }) {
  const loc = a.localization;
  return (
    <div className="grid gap-4 sm:grid-cols-2">
      <Card className="p-4">
        <p className="flex items-center gap-2 text-sm font-semibold"><Cpu size={16} className="text-brand" /> Model</p>
        <KeyValue rows={[
          ["Model", a.model.name],
          ["Checkpoint", <span className="font-mono text-[11px]" key="c">{a.model.checkpoint_id}</span>],
          ["Attention guidance", a.model.attention_guided ? "yes" : "no"],
          ["Input", `${a.input.kind?.toUpperCase()} ${a.input.width}×${a.input.height}`],
        ]} />
      </Card>
      <Card className="p-4">
        <p className="flex items-center gap-2 text-sm font-semibold"><Crosshair size={16} className="text-brand" /> Localization</p>
        {loc.available ? (
          <>
            <KeyValue rows={[["Peak value", fmt(loc.peak_probability, 2)], ["Area ≥ 0.5", pct(loc.area_fraction_above_0_5)]]} />
            <p className="mb-1 mt-2 text-xs text-navy-500">Where the localization map is concentrated</p>
            <ZoneBars zones={loc.top_zones} />
          </>
        ) : (
          <p className="mt-2 text-xs text-navy-500">{loc.reason}</p>
        )}
      </Card>
    </div>
  );
}

export default function Analysis() {
  const { trainedModels, selectedModel, models, modelsError, refreshModels, lastFile, setLastFile, lastAnalysis, setLastAnalysis, apiReachable } = useApp();
  const [file, setFile] = useState(lastFile);
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [result, setResult] = useState(lastAnalysis);
  const [tab, setTab] = useState("original");

  useEffect(() => setLastFile(file), [file, setLastFile]);
  const noModels = models && !trainedModels.length;

  const analyse = async () => {
    setBusy(true);
    setError(null);
    setProgress({ phase: "upload", fraction: 0 });
    try {
      const res = await api.analyze(file, selectedModel, setProgress);
      setResult(res);
      setLastAnalysis(res);
      setTab(res.images.localization ? "localization" : "gradcam");
    } catch (e) {
      setError(e);
      if (e.code === "model_not_trained") refreshModels();
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const tabs = result
    ? [
        { value: "original", label: "Original X-ray" },
        { value: "localization", label: "Localization heatmap", disabled: !result.images.localization },
        { value: "gradcam", label: "Grad-CAM++" },
        { value: "combined", label: "Combined overlay", disabled: !result.images.localization },
      ]
    : [];
  const visible = { original: [], localization: ["localization"], gradcam: ["gradcam"], combined: ["localization", "gradcam"] }[tab];

  return (
    <div>
      <PageHeader
        title="X-ray Analysis"
        subtitle="Upload a frontal chest X-ray to see whether the model detects a pneumonia-related opacity pattern, how certain it is, and where it looked."
        actions={result && (
          <>
            <Link to="/explainability"><Button variant="secondary" icon={Eye}>Explain in detail</Button></Link>
            <a href={urls.analysisReport(result.analysis_id)} target="_blank" rel="noreferrer">
              <Button variant="blue" icon={FileText}>Generate report</Button>
            </a>
          </>
        )}
      />
      {modelsError && <div className="mb-4"><ErrorState error={modelsError} onRetry={refreshModels} /></div>}
      {noModels && <div className="mb-6"><NoTrainedModelNotice /></div>}

      <div className="grid gap-6 lg:grid-cols-[minmax(0,380px)_minmax(0,1fr)]">
        <div className="space-y-4">
          <Card>
            <CardHeader icon={ScanLine} title="Input image" subtitle="PNG, JPEG or DICOM" />
            <div className="space-y-4 p-5">
              <Dropzone file={file} onFile={(f) => { setFile(f); setError(null); }} disabled={busy} />
              <div className="text-xs text-navy-500">
                Model: <span className="font-medium text-navy">{trainedModels.find((m) => m.key === selectedModel)?.name || "none available"}</span>
                {trainedModels.length > 1 && " (change it in the top bar)"}
              </div>
              <Button className="w-full" icon={ScanLine} onClick={analyse} loading={busy}
                disabled={!file || !selectedModel || !apiReachable || busy}>
                {busy ? "Analysing…" : "Analyse X-ray"}
              </Button>
              {progress && (
                <ProgressBar value={progress.phase === "upload" ? progress.fraction : 1}
                  label={progress.phase === "upload" ? `Uploading… ${Math.round(progress.fraction * 100)}%` : "Running model, uncertainty and Grad-CAM++…"} />
              )}
              {!apiReachable && <p className="text-xs text-rose-700">The analysis server is not reachable.</p>}
              <ErrorState error={error} compact />
            </div>
          </Card>
        </div>

        <div className="min-w-0 space-y-4">
          {busy && !result && <Card><Spinner label="Analysing the image…" /></Card>}
          {!result && !busy && (
            <Card>
              <EmptyState icon={ScanLine} title="No analysis yet">
                {noModels ? "Analysis becomes available once a trained model checkpoint exists." : "Choose an image on the left and press “Analyse X-ray”."}
              </EmptyState>
            </Card>
          )}
          {result && (
            <>
              <OutcomeCard a={result} />
              <StatusCards a={result} />
              <ModelAndLocalization a={result} />
            </>
          )}
        </div>
      </div>

      {result && (
        <Card className="mt-6">
          <CardHeader icon={Eye} title="Visualizations" subtitle="Coloured overlays are graded model outputs - they do not mark exact disease boundaries."
            action={<span className="hidden font-mono text-[11px] text-navy-300 sm:block">ID {result.analysis_id.slice(0, 8)}</span>} />
          <div className="grid gap-6 p-5 lg:grid-cols-[minmax(0,1fr)_300px]">
            <div className="min-w-0">
              <Tabs tabs={tabs} value={tab} onChange={setTab} />
              <div className="mx-auto mt-4 max-w-[640px]">
                <ImageViewer base={result.images.original} layers={result.images} visible={visible} size={result.image_size} label="Preprocessed chest X-ray" />
              </div>
            </div>
            <div className="space-y-3 text-[13px] text-navy-700">
              {tab === "original" && <p>The X-ray after the same preprocessing the model uses (contrast normalisation and resizing to {result.image_size} px).</p>}
              {(tab === "localization" || tab === "combined") && (
                <Notice tone="teal" title="Localization heatmap">{result.map_descriptions.localization}</Notice>
              )}
              {(tab === "gradcam" || tab === "combined") && (
                <Notice tone="info" title="Grad-CAM++">{result.map_descriptions.gradcam}</Notice>
              )}
              <p className="mb-1 font-medium text-navy">Grad-CAM++ is concentrated in</p>
              <ZoneBars zones={result.gradcam_top_zones} color="#d9480f" />
              <p className="text-[11px] text-navy-300">Zones use image coordinates; on a standard PA view the image-left side shows the patient's right lung.</p>
            </div>
          </div>
        </Card>
      )}
      {result && (
        <div className="mt-4">
          <Collapsible title="Technical details (for researchers)">
            <pre className="max-h-80 overflow-auto rounded bg-canvas p-3 text-[11px]">
              {JSON.stringify({ ...result, images: Object.keys(result.images) }, null, 2)}
            </pre>
          </Collapsible>
        </div>
      )}
    </div>
  );
}
