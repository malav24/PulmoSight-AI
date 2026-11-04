import { useEffect, useState } from "react";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { CheckCircle2, CircleDashed, Cpu, GitCompare, PlayCircle, RefreshCw } from "lucide-react";
import { useApp } from "../context/AppContext";
import { api } from "../services/api";
import { Badge, Button, Card, CardHeader, Collapsible, EmptyState, ErrorState, KeyValue, PageHeader, Spinner } from "../components/ui";
import { fmt, SERIES_COLORS } from "../lib/format";

const VAL_LABELS = {
  val_auc: "Validation AUC", val_dice_pos: "Validation Dice (positives)", val_attn_mass_in_box: "Attention mass in box",
  val_loss: "Validation loss", val_threshold_youden: "Validation threshold",
};

function ModelCard({ m, onUse, busyKey, isSelected }) {
  const ck = m.checkpoint;
  const val = ck.validation_metrics || {};
  const ev = m.evaluation;
  return (
    <Card className="flex flex-col">
      <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-4">
        <div className="min-w-0">
          <p className="text-[15px] font-semibold">{m.name}</p>
          <p className="text-xs text-navy-500">{m.short} · {m.parameters_millions} M parameters</p>
        </div>
        {m.trained ? <Badge tone="good" icon={CheckCircle2}>Trained</Badge>
          : ck.status === "invalid" ? <Badge tone="bad">Invalid checkpoint</Badge>
          : <Badge icon={CircleDashed}>Not trained</Badge>}
      </div>
      <div className="flex-1 space-y-3 px-5 py-4 text-[13px]">
        <p className="text-navy-700">{m.description}</p>
        <KeyValue rows={[
          ["Architecture", <span key="a" className="text-right text-xs">{m.architecture}</span>],
          ["Training objective", m.training_objective],
          ["Dataset", "RSNA Pneumonia Detection (train/val split)"],
          ["Checkpoint", ck.exists ? <span key="c" className="font-mono text-[11px]">{ck.checkpoint_id || "unreadable"}</span> : <span key="c" className="text-navy-300">not found</span>],
          m.trained && ["Saved", `${ck.saved_at?.replace("T", " ")} · epoch ${ck.epoch} · ${ck.size_mb} MB`],
          m.trained && ["Calibration", !m.calibration ? "not fitted" : m.calibration.matches_checkpoint ? `T = ${fmt(m.calibration.temperature, 2)}${m.calibration.temperature_at_bound ? " (at bound - unreliable)" : ""}` : "stale (other checkpoint)"],
        ]} />
        <div className="rounded-lg bg-canvas p-3">
          <p className="text-xs font-semibold text-navy-500">Validation metrics (best epoch)</p>
          {m.trained && Object.keys(val).length ? (
            <KeyValue rows={Object.entries(VAL_LABELS).filter(([k]) => k in val).map(([k, l]) => [l, fmt(val[k])])} />
          ) : (
            <p className="mt-1 text-xs text-navy-500">{m.trained ? "No validation metrics stored in this checkpoint." : "None - this model has not been trained. The architecture exists in code only."}</p>
          )}
        </div>
        {ev && (
          <div className="rounded-lg bg-canvas p-3">
            <p className="text-xs font-semibold text-navy-500">Test evaluation (RSNA test split)</p>
            {ev.stale && <p className="mt-1 text-xs font-medium text-amber-700">These figures belong to an earlier checkpoint - re-run the evaluation.</p>}
            <KeyValue rows={[["Test AUC", fmt(ev.rsna_test_auc)], ["Sensitivity / specificity", `${fmt(ev.sensitivity)} / ${fmt(ev.specificity)}`],
              ev.localization_dice !== null && ev.localization_dice !== undefined && ["Localization Dice", fmt(ev.localization_dice)],
              ev.nih_auc !== null && ev.nih_auc !== undefined && ["NIH external AUC", fmt(ev.nih_auc)]]} />
          </div>
        )}
      </div>
      <div className="flex items-center justify-between gap-2 border-t border-line px-5 py-3">
        <span className="text-xs text-navy-500">{m.active ? "Loaded on server" : m.loaded ? "In memory" : ""}</span>
        <Button variant={isSelected ? "secondary" : "primary"} icon={PlayCircle} disabled={!m.trained} loading={busyKey === m.key} onClick={() => onUse(m.key)}>
          {isSelected ? "Selected for analysis" : "Use for analysis"}
        </Button>
      </div>
    </Card>
  );
}

function CompareAB({ a, b, comparison }) {
  if (!a || !b) return null;
  const bothTrained = a.trained && b.trained;
  const rows = [
    ["Validation AUC", a.checkpoint.validation_metrics?.val_auc, b.checkpoint.validation_metrics?.val_auc],
    ["Validation Dice (positives)", a.checkpoint.validation_metrics?.val_dice_pos, b.checkpoint.validation_metrics?.val_dice_pos],
    ["Validation attention mass in box", a.checkpoint.validation_metrics?.val_attn_mass_in_box, b.checkpoint.validation_metrics?.val_attn_mass_in_box],
    ["Test AUC", a.evaluation?.rsna_test_auc, b.evaluation?.rsna_test_auc],
    ["Test localization Dice", a.evaluation?.localization_dice, b.evaluation?.localization_dice],
    ["Test attention mass in box", a.evaluation?.attention_mass_in_box, b.evaluation?.attention_mass_in_box],
    ["ECE (temperature-scaled)", a.evaluation?.ece_temperature_scaled, b.evaluation?.ece_temperature_scaled],
    ["NIH external AUC", a.evaluation?.nih_auc, b.evaluation?.nih_auc],
  ];
  const chart = rows.filter(([name, x, y]) => typeof x === "number" && typeof y === "number" && !/ECE/.test(name))
    .map(([name, x, y]) => ({ name, "Model A": x, "Model B": y }));
  const delong = comparison?.model_b_vs_model_a || comparison?.model_a_vs_model_b;
  return (
    <Card>
      <CardHeader icon={GitCompare} title="Model A vs Model B" subtitle="Same architecture; Model B adds attention guidance. Differences isolate the effect of guidance." />
      <div className="p-5">
        {!bothTrained ? (
          <EmptyState icon={GitCompare} title="Comparison not available yet">
            Both models must be trained. Status: Model A {a.trained ? "trained" : "not trained"}, Model B {b.trained ? "trained" : "not trained"}.
          </EmptyState>
        ) : (
          <div className="grid gap-6 lg:grid-cols-2">
            <div className="overflow-x-auto">
              <table className="w-full text-[13px]">
                <thead className="text-left text-xs text-navy-500"><tr><th className="pb-2">Metric</th><th>Model A</th><th>Model B</th></tr></thead>
                <tbody className="divide-y divide-line">
                  {rows.map(([n, x, y]) => <tr key={n}><td className="py-2">{n}</td><td>{fmt(x)}</td><td>{fmt(y)}</td></tr>)}
                </tbody>
              </table>
              {delong ? (
                <p className="mt-3 text-xs text-navy-500">DeLong paired test on RSNA test: ΔAUC {fmt(delong.difference)}, p = {fmt(delong.p_value, 4)}{delong.p_value_holm !== undefined ? ` (Holm-adjusted ${fmt(delong.p_value_holm, 4)})` : ""}.</p>
              ) : <p className="mt-3 text-xs text-navy-500">No paired statistical test has been run yet (Research Experiments → compare models).</p>}
            </div>
            {chart.length ? (
              <div className="h-72">
                <ResponsiveContainer>
                  <BarChart data={chart} layout="vertical" margin={{ left: 40 }}>
                    <CartesianGrid strokeDasharray="3 3" stroke="#e3e8ef" />
                    <XAxis type="number" domain={[0, 1]} tick={{ fontSize: 11 }} />
                    <YAxis type="category" dataKey="name" width={150} tick={{ fontSize: 11 }} />
                    <Tooltip formatter={(v) => fmt(v)} />
                    <Legend wrapperStyle={{ fontSize: 12 }} />
                    <Bar dataKey="Model A" fill={SERIES_COLORS.model_a} radius={[0, 4, 4, 0]} />
                    <Bar dataKey="Model B" fill={SERIES_COLORS.model_b} radius={[0, 4, 4, 0]} />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            ) : <EmptyState title="No shared numeric metrics to chart" />}
          </div>
        )}
      </div>
    </Card>
  );
}

export default function ModelLab() {
  const { models, modelsError, refreshModels, selectedModel, setSelectedModel, refreshHealth } = useApp();
  const [busyKey, setBusyKey] = useState(null);
  const [error, setError] = useState(null);
  const [comparison, setComparison] = useState(null);

  useEffect(() => {
    api.experimentResults().then((r) => setComparison(r.comparisons?.rsna_test || null)).catch(() => {});
  }, []);

  const use = async (key) => {
    setBusyKey(key);
    setError(null);
    try {
      await api.loadModel(key);
      setSelectedModel(key);
      await refreshModels();
      refreshHealth();
    } catch (e) {
      setError(e);
    } finally {
      setBusyKey(null);
    }
  };

  if (modelsError) return <ErrorState error={modelsError} onRetry={refreshModels} />;
  if (!models) return <Spinner label="Loading models…" />;
  const by = Object.fromEntries(models.models.map((m) => [m.key, m]));
  return (
    <div className="space-y-6">
      <PageHeader title="Model Laboratory" subtitle="Every model in the study, with its real training status. A model counts as trained only when a valid checkpoint file exists."
        actions={<Button variant="secondary" icon={RefreshCw} onClick={refreshModels}>Refresh</Button>} />
      {!models.any_trained && (
        <Card className="p-5 text-sm"><p className="font-semibold">Models not trained yet.</p>
          <p className="mt-1 text-navy-500">All architectures are implemented, but no checkpoints exist. See docs/TRAINING_GUIDE.md for training commands.</p></Card>
      )}
      <ErrorState error={error} />
      <div className="grid gap-5 md:grid-cols-2 xl:grid-cols-3">
        {models.primary.map((k) => by[k] && <ModelCard key={k} m={by[k]} onUse={use} busyKey={busyKey} isSelected={selectedModel === k} />)}
      </div>
      <CompareAB a={by.model_a} b={by.model_b} comparison={comparison} />
      {by.ablation_cls_only && (
        <Collapsible title="Research control: classification-only ablation">
          <div className="max-w-md"><ModelCard m={by.ablation_cls_only} onUse={use} busyKey={busyKey} isSelected={selectedModel === "ablation_cls_only"} /></div>
        </Collapsible>
      )}
      <Card className="p-5 text-[13px] text-navy-700">
        <p className="flex items-center gap-2 font-semibold text-navy"><Cpu size={16} /> Loading your own checkpoint</p>
        <p className="mt-1">Copy a checkpoint produced by this project's training scripts to the expected location (e.g. <code>checkpoints/hybrid/model_b_best.pth</code>), press Refresh, then “Use for analysis”. Files from other projects are rejected because the stored model name and weights must match exactly.</p>
      </Card>
    </div>
  );
}
