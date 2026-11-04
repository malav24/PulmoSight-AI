import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { ArrowRight, BarChart3, Clock, Cpu, Database, FileText, ScanLine, Server } from "lucide-react";
import { useApp } from "../context/AppContext";
import { api, urls } from "../services/api";
import { Badge, Button, Card, CardHeader, EmptyState, ErrorState, Spinner } from "../components/ui";
import { fmt } from "../lib/format";

function useLoad(fn) {
  const [state, setState] = useState({ data: null, error: null, loading: true });
  const load = () => {
    setState((s) => ({ ...s, loading: true }));
    fn().then((data) => setState({ data, error: null, loading: false })).catch((error) => setState({ data: null, error, loading: false }));
  };
  useEffect(load, []); // eslint-disable-line react-hooks/exhaustive-deps
  return { ...state, reload: load };
}

export default function Overview() {
  const { models, modelsError, refreshModels, trainedModels, health } = useApp();
  const ds = useLoad(() => api.datasets());
  const recent = useLoad(() => api.analyses(6));
  const results = useLoad(() => api.experimentResults());

  const evaluated = results.data?.available
    ? Object.values(results.data.models).filter((m) => m.metrics?.rsna_test)
    : [];

  return (
    <div className="space-y-6">
      <Card className="overflow-hidden">
        <div className="grid gap-6 p-6 sm:p-8 lg:grid-cols-[1.4fr_1fr]">
          <div>
            <Badge tone="info">Academic research project</Badge>
            <h1 className="mt-3 text-2xl font-bold tracking-tight text-navy sm:text-3xl">RIGHT FOR THE RIGHT REASON</h1>
            <p className="mt-1 text-base font-medium text-teal-600">Explanation-Guided Pneumonia Detection and Localization</p>
            <p className="mt-4 max-w-2xl text-sm leading-relaxed text-navy-700">
              AI models can reach the right answer on chest X-rays for the wrong reasons - for example by reacting to text markers or
              image borders instead of the lungs. This project tests whether guiding a model with radiologists' annotations during
              training makes it look at the right regions, and whether that improves reliability and generalisation.
            </p>
            <div className="mt-5 flex flex-wrap gap-2">
              <Link to="/analysis"><Button icon={ScanLine}>Analyse an X-ray</Button></Link>
              <Link to="/about"><Button variant="secondary">About the research</Button></Link>
            </div>
          </div>
          <div className="rounded-xl bg-canvas p-5 text-[13px] text-navy-700">
            <p className="font-semibold text-navy">The study in brief</p>
            <ul className="mt-2 space-y-2">
              <li><b>Model A</b> - hybrid CNN-Transformer that classifies and localizes.</li>
              <li><b>Model B</b> - the same network, additionally guided by RSNA radiologist boxes.</li>
              <li><b>Baselines</b> - ResNet50, DenseNet121, EfficientNet-B0, ViT-Small.</li>
              <li><b>Evaluation</b> - accuracy, localization, explanation quality, shortcut tests, NIH external data, calibration.</li>
            </ul>
          </div>
        </div>
      </Card>

      <div className="grid gap-4 md:grid-cols-3">
        <Card className="p-5">
          <p className="flex items-center gap-2 text-sm font-semibold"><Cpu size={16} className="text-brand" /> Model status</p>
          {modelsError ? <div className="mt-3"><ErrorState error={modelsError} onRetry={refreshModels} compact /></div> : !models ? <Spinner /> : (
            trainedModels.length ? (
              <>
                <p className="mt-3 text-2xl font-semibold">{trainedModels.length} <span className="text-sm font-normal text-navy-500">of {models.models.length} trained</span></p>
                <div className="mt-3 flex flex-wrap gap-1.5">{trainedModels.map((m) => <Badge key={m.key} tone="good">{m.name}</Badge>)}</div>
              </>
            ) : (
              <>
                <p className="mt-3 text-lg font-semibold text-navy">Models not trained yet.</p>
                <p className="mt-1 text-xs text-navy-500">No checkpoints were found. Analysis stays disabled until a trained model is available.</p>
              </>
            )
          )}
          <Link to="/models" className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-brand">Model Laboratory <ArrowRight size={12} /></Link>
        </Card>
        <Card className="p-5">
          <p className="flex items-center gap-2 text-sm font-semibold"><Database size={16} className="text-brand" /> Datasets</p>
          {ds.error ? <div className="mt-3"><ErrorState error={ds.error} onRetry={ds.reload} compact /></div> : ds.loading ? <Spinner /> : (
            <ul className="mt-3 space-y-2 text-[13px]">
              <li className="flex items-center justify-between">RSNA (training)
                <Badge tone={ds.data.rsna.ready_for_training ? "good" : "neutral"}>{ds.data.rsna.ready_for_training ? `${ds.data.rsna.image_count.toLocaleString()} images` : "not found"}</Badge></li>
              <li className="flex items-center justify-between">NIH (external)
                <Badge tone={ds.data.nih.ready_for_external_evaluation ? "good" : "neutral"}>{ds.data.nih.ready_for_external_evaluation ? `${ds.data.nih.image_count.toLocaleString()} images` : "not found"}</Badge></li>
            </ul>
          )}
          <Link to="/datasets" className="mt-4 inline-flex items-center gap-1 text-xs font-medium text-brand">Dataset Management <ArrowRight size={12} /></Link>
        </Card>
        <Card className="p-5">
          <p className="flex items-center gap-2 text-sm font-semibold"><Server size={16} className="text-brand" /> System</p>
          {!health ? <Spinner label="Contacting server…" /> : (
            <ul className="mt-3 space-y-1.5 text-[13px] text-navy-700">
              <li>Server: <b>{health.status === "ok" ? "online" : health.status}</b></li>
              <li>Compute: <b>{health.ml?.device?.toUpperCase() ?? "—"}</b>{health.ml?.cuda_available ? " (GPU)" : ""}</li>
              <li>Active model: <b>{models?.models?.find((m) => m.key === health.models.active)?.name || "none loaded"}</b></li>
            </ul>
          )}
        </Card>
      </div>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader icon={BarChart3} title="Experiment performance" subtitle="Read from evaluation output files" />
          {results.loading ? <Spinner /> : results.error ? <div className="p-5"><ErrorState error={results.error} onRetry={results.reload} /></div> : evaluated.length ? (
            <div className="overflow-x-auto p-5">
              <table className="w-full text-[13px]">
                <thead className="text-left text-xs text-navy-500"><tr><th className="pb-2">Model</th><th>RSNA test AUC</th><th>Sensitivity</th><th>Specificity</th><th>NIH AUC</th></tr></thead>
                <tbody className="divide-y divide-line">
                  {evaluated.map((m) => {
                    const c = m.metrics.rsna_test.classification;
                    return (
                      <tr key={m.key}>
                        <td className="py-2 font-medium">{m.name}</td><td>{fmt(c.roc_auc)}</td><td>{fmt(c.sensitivity)}</td>
                        <td>{fmt(c.specificity)}</td><td>{fmt(m.metrics.nih_external?.classification?.roc_auc)}</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
              <Link to="/experiments" className="mt-3 inline-flex items-center gap-1 text-xs font-medium text-brand">All experiments <ArrowRight size={12} /></Link>
            </div>
          ) : (
            <EmptyState icon={BarChart3} title="No experiment results available">Performance figures appear here once models have been trained and evaluated.</EmptyState>
          )}
        </Card>
        <Card>
          <CardHeader icon={Clock} title="Recent analyses" subtitle="Stored on the server without the original file" />
          {recent.loading ? <Spinner /> : recent.error ? <div className="p-5"><ErrorState error={recent.error} onRetry={recent.reload} /></div> : recent.data.analyses.length ? (
            <ul className="divide-y divide-line">
              {recent.data.analyses.map((a) => (
                <li key={a.analysis_id} className="flex items-center justify-between gap-3 px-5 py-3 text-[13px]">
                  <div className="min-w-0">
                    <p className="truncate font-medium">{a.outcome === "suspected" ? "Opacity pattern suspected" : a.outcome === "not_detected" ? "No opacity pattern detected" : "Abstained"}</p>
                    <p className="text-xs text-navy-500">{a.model_name} · score {fmt(a.probability, 2)} · {a.created_at?.replace("T", " ")}</p>
                  </div>
                  <a href={urls.analysisReport(a.analysis_id)} target="_blank" rel="noreferrer" className="inline-flex shrink-0 items-center gap-1 text-xs font-medium text-brand">
                    <FileText size={13} /> Report
                  </a>
                </li>
              ))}
            </ul>
          ) : (
            <EmptyState icon={Clock} title="No analyses yet" action={<Link to="/analysis"><Button variant="secondary" icon={ScanLine}>Analyse an X-ray</Button></Link>} />
          )}
        </Card>
      </div>
    </div>
  );
}
