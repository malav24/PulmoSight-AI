import { useCallback, useEffect, useRef, useState } from "react";
import {
  Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts";
import {
  Activity, BarChart3, Crosshair, Download, FlaskConical, Globe2, Lightbulb, Play, ShieldAlert, Sigma, Target, TrendingUp,
} from "lucide-react";
import { useApp } from "../context/AppContext";
import { api, urls } from "../services/api";
import { Badge, Button, Card, CardHeader, Collapsible, EmptyState, ErrorState, Notice, PageHeader, Spinner } from "../components/ui";
import { fmt, MODEL_ORDER, SERIES_COLORS } from "../lib/format";

const chartTick = { fontSize: 11, fill: "#4a6380" };

function Panel({ letter, icon, title, subtitle, has, children, note }) {
  return (
    <Card>
      <CardHeader icon={icon} title={`${letter}. ${title}`} subtitle={subtitle} />
      <div className="p-5">
        {has ? children : <EmptyState title="No experiment results available">{note || "Run an evaluation to populate this panel."}</EmptyState>}
      </div>
    </Card>
  );
}

function Table({ head, rows }) {
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[480px] text-[13px]">
        <thead className="text-left text-xs text-navy-500"><tr>{head.map((h) => <th key={h} className="pb-2 pr-3">{h}</th>)}</tr></thead>
        <tbody className="divide-y divide-line">{rows.map((r, i) => <tr key={i}>{r.map((c, j) => <td key={j} className="py-2 pr-3">{c}</td>)}</tr>)}</tbody>
      </table>
    </div>
  );
}

const ci = (o) => (o && o.ci_low !== null && o.ci_low !== undefined ? `${fmt(o.ci_low)}–${fmt(o.ci_high)}` : "—");

function GroupedBars({ data, keys, height = 260, domain = [0, 1] }) {
  return (
    <div style={{ height }}>
      <ResponsiveContainer>
        <BarChart data={data}>
          <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" />
          <XAxis dataKey="name" tick={chartTick} interval={0} />
          <YAxis domain={domain} tick={chartTick} />
          <Tooltip formatter={(v) => fmt(v)} />
          <Legend wrapperStyle={{ fontSize: 12 }} />
          {keys.map((k) => <Bar key={k.key} dataKey={k.key} name={k.name} fill={k.color || SERIES_COLORS[k.key] || "#64748b"} radius={[4, 4, 0, 0]} />)}
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}

function pivot(models, metricDefs, getter) {
  // rows = metrics, series = models
  return metricDefs.map(([label, path]) => {
    const row = { name: label };
    models.forEach((m) => { const v = getter(m, path); if (typeof v === "number") row[m.key] = v; });
    return row;
  });
}

const get = (obj, path) => path.split(".").reduce((o, k) => (o == null ? undefined : o[k]), obj);

function JobRunner({ trained, onFinished }) {
  const [kind, setKind] = useState("evaluate");
  const [model, setModel] = useState(trained[0]?.key || "");
  const [opts, setOpts] = useState({ nih: false, stress: true, fit_ood: true, explain_n: 32, epochs: "", confirm: false });
  const [job, setJob] = useState(null);
  const [error, setError] = useState(null);
  const timer = useRef(null);

  useEffect(() => () => clearInterval(timer.current), []);
  useEffect(() => { if (!model && trained[0]) setModel(trained[0].key); }, [trained, model]);

  const poll = (id) => {
    clearInterval(timer.current);
    timer.current = setInterval(async () => {
      try {
        const j = await api.job(id);
        setJob(j);
        if (j.status !== "running") { clearInterval(timer.current); onFinished(); }
      } catch (e) { setError(e); clearInterval(timer.current); }
    }, 2000);
  };

  const start = async () => {
    setError(null);
    const params = kind === "evaluate" ? { nih: opts.nih, stress: opts.stress, fit_ood: opts.fit_ood, explain_n: Number(opts.explain_n) }
      : kind === "train" ? { confirm: opts.confirm, ...(opts.epochs ? { epochs: Number(opts.epochs) } : {}) }
      : kind === "compare_models" ? { models: MODEL_ORDER.filter((k) => trained.some((t) => t.key === k)), dataset: "rsna_test" }
      : {};
    try {
      const j = await api.runExperiment(kind, ["evaluate", "train"].includes(kind) ? model : null, params);
      setJob(j);
      poll(j.job_id);
    } catch (e) { setError(e); }
  };

  const needsModel = ["evaluate", "train"].includes(kind);
  const running = job?.status === "running";
  return (
    <Card>
      <CardHeader icon={FlaskConical} title="Run an experiment" subtitle="Runs as a separate background process; analysis keeps working meanwhile. Nothing starts automatically." />
      <div className="grid gap-5 p-5 lg:grid-cols-[320px_minmax(0,1fr)]">
        <div className="space-y-3 text-[13px]">
          <label className="block">Experiment
            <select value={kind} onChange={(e) => setKind(e.target.value)} className="mt-1 w-full rounded-lg border border-line px-2 py-2">
              <option value="evaluate">Evaluate a trained model</option>
              <option value="compare_models">Compare evaluated models (DeLong)</option>
              <option value="ablation_summary">Summarise ablation study</option>
              <option value="prepare_rsna">Prepare RSNA splits & statistics</option>
              <option value="prepare_nih">Prepare NIH external set</option>
              <option value="train">Train a model (long)</option>
            </select>
          </label>
          {needsModel && (
            <label className="block">Model
              <select value={model} onChange={(e) => setModel(e.target.value)} className="mt-1 w-full rounded-lg border border-line px-2 py-2">
                {(kind === "train" ? MODEL_ORDER.map((k) => ({ key: k, name: k })) : trained).map((m) => <option key={m.key} value={m.key}>{m.name}</option>)}
              </select>
            </label>
          )}
          {kind === "evaluate" && (
            <div className="space-y-1.5">
              {[["stress", "Shortcut stress tests"], ["nih", "NIH external evaluation"], ["fit_ood", "Fit out-of-distribution statistics"]].map(([k, l]) => (
                <label key={k} className="flex items-center gap-2"><input type="checkbox" checked={opts[k]} onChange={(e) => setOpts({ ...opts, [k]: e.target.checked })} />{l}</label>
              ))}
              <label className="flex items-center gap-2">Grad-CAM++ images
                <input type="number" min={0} max={500} value={opts.explain_n} onChange={(e) => setOpts({ ...opts, explain_n: e.target.value })} className="w-20 rounded border border-line px-2 py-1" />
              </label>
            </div>
          )}
          {kind === "train" && (
            <div className="space-y-2">
              <label className="flex items-center gap-2">Epochs (optional)
                <input type="number" min={1} value={opts.epochs} onChange={(e) => setOpts({ ...opts, epochs: e.target.value })} className="w-20 rounded border border-line px-2 py-1" />
              </label>
              <label className="flex items-start gap-2"><input type="checkbox" className="mt-0.5" checked={opts.confirm} onChange={(e) => setOpts({ ...opts, confirm: e.target.checked })} />
                I understand training can take hours and needs a GPU.</label>
            </div>
          )}
          <Button icon={Play} onClick={start} loading={running} disabled={running || (needsModel && !model) || (kind === "train" && !opts.confirm)}>Start</Button>
          {needsModel && kind === "evaluate" && !trained.length && <p className="text-xs text-navy-500">Evaluation needs a trained model.</p>}
          <ErrorState error={error} compact />
        </div>
        <div className="min-w-0">
          {job ? (
            <div>
              <div className="flex flex-wrap items-center gap-2 text-[13px]">
                <Badge tone={job.status === "succeeded" ? "good" : job.status === "failed" ? "bad" : "info"}>{job.status}</Badge>
                <code className="truncate text-xs text-navy-500">{job.command}</code>
              </div>
              <pre className="mt-2 max-h-64 overflow-auto rounded-lg bg-navy p-3 text-[11px] leading-relaxed text-slate-100">{(job.log_tail || []).join("\n") || "Starting…"}</pre>
            </div>
          ) : <EmptyState icon={FlaskConical} title="No experiment started in this session" />}
        </div>
      </div>
    </Card>
  );
}

export default function Experiments() {
  const { trainedModels } = useApp();
  const [res, setRes] = useState(null);
  const [error, setError] = useState(null);
  const [calModel, setCalModel] = useState(null);
  const [histModel, setHistModel] = useState(null);

  const load = useCallback(() => {
    api.experimentResults().then((r) => { setRes(r); setError(null); }).catch(setError);
  }, []);
  useEffect(load, [load]);

  if (error) return <ErrorState error={error} onRetry={load} />;
  if (!res) return <Spinner label="Loading experiment results…" />;

  const all = MODEL_ORDER.map((k) => res.models[k]).filter(Boolean);
  const evald = all.filter((m) => m.metrics?.rsna_test);
  const T = (m) => m.metrics.rsna_test;
  const keysOf = (ms) => ms.map((m) => ({ key: m.key, name: m.name }));

  const withLoc = evald.filter((m) => T(m).localization_heatmap);
  const withExpl = evald.filter((m) => T(m).explanations?.gradcam_faithfulness || T(m).attention_alignment);
  const withStress = evald.filter((m) => T(m).stress_tests);
  const withNih = evald.filter((m) => m.metrics.generalization);
  const withCurves = all.filter((m) => m.curves?.rsna_test?.roc);
  const comps = res.comparisons || {};
  const cal = res.models[calModel] || evald.find((m) => m.curves?.rsna_test) || null;
  const histCandidates = all.filter((m) => m.training_log?.length);
  const hist = res.models[histModel] || histCandidates[0] || null;

  const stressTests = ["outside_lung_masking", "border_masking", "marker_masking", "background_perturbation"];
  const stressLabels = { outside_lung_masking: "Outside thorax", border_masking: "Border", marker_masking: "Markers", background_perturbation: "Background noise" };

  return (
    <div className="space-y-6">
      <PageHeader title="Research Experiments" subtitle="All figures are read from evaluation output files on the server. Panels stay empty until the corresponding experiment has been run."
        actions={<Button variant="secondary" onClick={load}>Reload results</Button>} />
      {!res.available && (
        <Notice tone="info" icon={BarChart3} title="No experiment results available">
          Train a model, then run “Evaluate a trained model” below (or <code>python -m ml.evaluation.evaluate --model model_b --nih --stress</code>).
        </Notice>
      )}

      <JobRunner trained={trainedModels} onFinished={load} />

      <Panel letter="A" icon={Target} title="Classification performance" subtitle="RSNA held-out test split · threshold chosen on validation (Youden J)" has={evald.length > 0}>
        <Table head={["Model", "AUC", "95% CI (bootstrap)", "Sensitivity", "Specificity", "Precision", "F1", "n"]}
          rows={evald.map((m) => { const c = T(m).classification; return [m.name, fmt(c.roc_auc), ci(c.roc_auc_bootstrap_ci), fmt(c.sensitivity), fmt(c.specificity), fmt(c.precision), fmt(c.f1), c.n]; })} />
        {withCurves.length > 0 && (
          <div className="mt-5 h-72">
            <ResponsiveContainer>
              <LineChart margin={{ right: 10 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" />
                <XAxis type="number" dataKey="fpr" domain={[0, 1]} tick={chartTick} label={{ value: "1 − specificity", position: "insideBottom", offset: -2, fontSize: 11 }} />
                <YAxis type="number" domain={[0, 1]} tick={chartTick} label={{ value: "Sensitivity", angle: -90, position: "insideLeft", fontSize: 11 }} />
                <Tooltip formatter={(v) => fmt(v)} />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#cbd5e1" strokeDasharray="4 4" />
                {withCurves.map((m) => <Line key={m.key} data={m.curves.rsna_test.roc} dataKey="tpr" name={m.name} dot={false} stroke={SERIES_COLORS[m.key]} strokeWidth={2} />)}
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
        <Collapsible title="Limitations">AUC ignores calibration and the operating point; sensitivity/specificity depend on the threshold; precision depends on prevalence. Labels are radiologist opacity annotations, not confirmed diagnoses.</Collapsible>
      </Panel>

      <Panel letter="B" icon={Crosshair} title="Localization performance" subtitle="Predicted heatmap (≥ 0.5) vs RSNA boxes, positive test images" has={withLoc.length > 0}
        note="Available for hybrid models with a trained localization head after evaluation.">
        <GroupedBars keys={keysOf(withLoc)} data={pivot(withLoc, [["IoU", "iou"], ["Dice", "dice"], ["Pointing game", "pointing_game"], ["Mass in box", "mass_in_box"]], (m, p) => T(m).localization_heatmap[p])} />
        <Table head={["Model", "IoU", "Dice", "Pointing game", "Mass in box", "n"]}
          rows={withLoc.map((m) => { const l = T(m).localization_heatmap; return [m.name, fmt(l.iou), fmt(l.dice), fmt(l.pointing_game), fmt(l.mass_in_box), l.n]; })} />
      </Panel>

      <Panel letter="C" icon={Lightbulb} title="Explanation quality" subtitle="Grad-CAM++ faithfulness (deletion ↓ / insertion ↑) and plausibility; hybrid attention alignment" has={withExpl.length > 0}>
        <Table head={["Model", "Deletion AUC ↓", "Insertion AUC ↑", "Grad-CAM++ pointing", "Grad-CAM++ mass in box", "Attention mass in box"]}
          rows={withExpl.map((m) => { const e = T(m).explanations || {}; return [m.name, fmt(e.gradcam_faithfulness?.deletion_auc), fmt(e.gradcam_faithfulness?.insertion_auc), fmt(e.gradcam_localization?.pointing_game), fmt(e.gradcam_localization?.mass_in_box), fmt(T(m).attention_alignment?.attention_mass_in_box)]; })} />
        <div className="mt-4"><GroupedBars keys={keysOf(withExpl)} data={pivot(withExpl, [["Deletion AUC", "explanations.gradcam_faithfulness.deletion_auc"], ["Insertion AUC", "explanations.gradcam_faithfulness.insertion_auc"], ["Pointing", "explanations.gradcam_localization.pointing_game"], ["Attention in box", "attention_alignment.attention_mass_in_box"]], (m, p) => get(T(m), p))} /></div>
        <Collapsible title="Limitations">Deletion/insertion perturb images out of distribution; plausibility (agreement with boxes) is not faithfulness.</Collapsible>
      </Panel>

      <Panel letter="D" icon={ShieldAlert} title="Shortcut-learning stress tests" subtitle="AUC drop when non-lung content is masked or perturbed (smaller is better)" has={withStress.length > 0}
        note="Run an evaluation with “Shortcut stress tests” enabled.">
        <GroupedBars keys={keysOf(withStress)} domain={["auto", "auto"]} data={stressTests.map((t) => { const row = { name: stressLabels[t] }; withStress.forEach((m) => { const v = T(m).stress_tests[t]?.auc_drop; if (typeof v === "number") row[m.key] = v; }); return row; })} />
        <Table head={["Model", "Test", "AUC", "AUC drop", "Mean |Δp|", "Decision flips"]}
          rows={withStress.flatMap((m) => stressTests.map((t) => { const s = T(m).stress_tests[t] || {}; return [m.name, stressLabels[t], fmt(s.auc), fmt(s.auc_drop), fmt(s.mean_abs_prob_change), fmt(s.decision_flip_rate)]; }))} />
        <p className="mt-2 text-xs text-navy-500">The thorax region is a fixed geometric ellipse, not a lung segmentation; marker detection is heuristic.</p>
      </Panel>

      <Panel letter="E" icon={Globe2} title="Cross-dataset generalization" subtitle="RSNA internal test vs NIH ChestX-ray14 external set" has={withNih.length > 0}
        note="Run an evaluation with “NIH external evaluation” enabled (needs the NIH dataset).">
        <GroupedBars keys={[{ key: "internal", name: "RSNA test", color: "#0e8f86" }, { key: "external", name: "NIH external", color: "#2f6fd6" }]}
          data={withNih.map((m) => ({ name: m.name, internal: m.metrics.generalization.internal_auc.estimate, external: m.metrics.generalization.external_auc.estimate }))} />
        <Table head={["Model", "RSNA AUC (CI)", "NIH AUC (CI)", "Difference (CI)", "NIH notes"]}
          rows={withNih.map((m) => { const g = m.metrics.generalization; const d = g.auc_difference_internal_minus_external; return [m.name, `${fmt(g.internal_auc.estimate)} (${ci(g.internal_auc)})`, `${fmt(g.external_auc.estimate)} (${ci(g.external_auc)})`, `${fmt(d.auc_difference)} (${ci(d)})`, (m.metrics.nih_external?.selection_notes || []).join(" ") || "—"]; })} />
        <p className="mt-2 text-xs text-navy-500">NIH labels are text-mined and defined differently; both datasets come from the same hospital - this is not multi-site validation.</p>
      </Panel>

      <Panel letter="F" icon={Activity} title="Calibration" subtitle="Temperature fitted on RSNA validation, applied unchanged to test" has={evald.length > 0}>
        <Table head={["Model", "Temperature", "ECE raw", "ECE scaled", "Brier raw", "Brier scaled"]}
          rows={evald.map((m) => [m.name, fmt(m.metrics.temperature, 2), fmt(T(m).calibration_uncalibrated?.ece), fmt(T(m).calibration_temperature_scaled?.ece), fmt(T(m).calibration_uncalibrated?.brier), fmt(T(m).calibration_temperature_scaled?.brier)])} />
        {cal?.curves?.rsna_test && (
          <div className="mt-5">
            <label className="text-[13px] text-navy-500">Reliability diagram for{" "}
              <select value={cal.key} onChange={(e) => setCalModel(e.target.value)} className="rounded border border-line px-2 py-1 text-navy">
                {evald.filter((m) => m.curves?.rsna_test).map((m) => <option key={m.key} value={m.key}>{m.name}</option>)}
              </select>
            </label>
            <div className="mt-2 h-72">
              <ResponsiveContainer>
                <LineChart>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" />
                  <XAxis type="number" dataKey="confidence" domain={[0, 1]} tick={chartTick} label={{ value: "Predicted probability", position: "insideBottom", offset: -2, fontSize: 11 }} />
                  <YAxis type="number" domain={[0, 1]} tick={chartTick} label={{ value: "Observed frequency", angle: -90, position: "insideLeft", fontSize: 11 }} />
                  <Tooltip formatter={(v) => fmt(v)} />
                  <Legend wrapperStyle={{ fontSize: 12 }} />
                  <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#cbd5e1" strokeDasharray="4 4" />
                  <Line data={cal.curves.rsna_test.reliability_raw} dataKey="observed" name="Uncalibrated" stroke="#94a3b8" strokeWidth={2} />
                  {cal.curves.rsna_test.reliability_calibrated && <Line data={cal.curves.rsna_test.reliability_calibrated} dataKey="observed" name="Temperature-scaled" stroke="#0e8f86" strokeWidth={2} />}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}
      </Panel>

      <Panel letter="G" icon={Sigma} title="Statistical analysis" subtitle="95% confidence intervals and paired DeLong tests (Holm-corrected)" has={evald.length > 0}>
        <Table head={["Model", "AUC", "Bootstrap 95% CI", "DeLong 95% CI"]}
          rows={evald.map((m) => { const c = T(m).classification; return [m.name, fmt(c.roc_auc), ci(c.roc_auc_bootstrap_ci), ci(c.roc_auc_delong_ci)]; })} />
        {Object.entries(comps).map(([ds, c]) => c && (
          <div key={ds} className="mt-5">
            <p className="mb-2 text-[13px] font-semibold">Paired comparisons on {ds === "rsna_test" ? "RSNA test" : "NIH"} ({c._n_paired_images} images)</p>
            <Table head={["Comparison", "AUC A", "AUC B", "Δ", "z", "p", "p (Holm)"]}
              rows={Object.entries(c).filter(([k]) => !k.startsWith("_")).map(([k, v]) => [k.replace(/_vs_/, " vs "), fmt(v.auc_a), fmt(v.auc_b), fmt(v.difference), fmt(v.z, 2), fmt(v.p_value, 4), fmt(v.p_value_holm, 4)])} />
          </div>
        ))}
        {!Object.keys(comps).length && <p className="mt-3 text-xs text-navy-500">No paired comparison yet - run “Compare evaluated models”.</p>}
      </Panel>

      <Card>
        <CardHeader icon={TrendingUp} title="Training history" subtitle="From outputs/logs/<model>_training_log.csv" />
        <div className="p-5">
          {hist ? (
            <>
              <select value={hist.key} onChange={(e) => setHistModel(e.target.value)} className="rounded border border-line px-2 py-1 text-[13px]">
                {histCandidates.map((m) => <option key={m.key} value={m.key}>{m.name}</option>)}
              </select>
              <div className="mt-3 h-64">
                <ResponsiveContainer>
                  <LineChart data={hist.training_log}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" />
                    <XAxis dataKey="epoch" tick={chartTick} />
                    <YAxis yAxisId="l" tick={chartTick} />
                    <YAxis yAxisId="r" orientation="right" domain={[0, 1]} tick={chartTick} />
                    <Tooltip formatter={(v) => fmt(v)} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Line yAxisId="l" dataKey="train_loss" name="Train loss" stroke="#94a3b8" />
                    <Line yAxisId="l" dataKey="val_loss" name="Val loss" stroke="#2f6fd6" />
                    <Line yAxisId="r" dataKey="val_auc" name="Val AUC" stroke="#0e8f86" strokeWidth={2} />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            </>
          ) : <EmptyState title="No training logs found" />}
        </div>
      </Card>

      <Card>
        <CardHeader icon={Download} title="Downloadable results" subtitle="Per-image predictions, metrics and reports (CSV / JSON)" />
        <div className="p-5">
          {res.files.length ? (
            <ul className="grid gap-2 sm:grid-cols-2 lg:grid-cols-3">
              {res.files.map((f) => (
                <li key={f.name}>
                  <a href={urls.resultFile(f.name)} className="flex items-center justify-between gap-2 rounded-lg border border-line px-3 py-2 text-[13px] hover:bg-canvas" download>
                    <span className="truncate">{f.name}</span><span className="shrink-0 text-xs text-navy-500">{f.size_kb} KB</span>
                  </a>
                </li>
              ))}
            </ul>
          ) : <EmptyState icon={Download} title="No result files yet" />}
        </div>
      </Card>
    </div>
  );
}
