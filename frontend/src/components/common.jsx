import { Link } from "react-router-dom";
import { Terminal } from "lucide-react";
import { Notice } from "./ui";
import { fmt } from "../lib/format";

export function NoTrainedModelNotice() {
  return (
    <Notice tone="warn" icon={Terminal} title="Models not trained yet - analysis is disabled">
      <p>No trained checkpoint is available, so the system cannot produce predictions. It will never show a made-up result.</p>
      <ol className="mt-2 list-decimal space-y-1 pl-5 text-[13px]">
        <li>Place the RSNA dataset in <code className="rounded bg-white/70 px-1">data/rsna/</code> (see <Link className="underline" to="/datasets">Dataset Management</Link>).</li>
        <li>Train a model, e.g. <code className="rounded bg-white/70 px-1">python -m ml.training.train_model_b</code>, or copy an existing checkpoint to <code className="rounded bg-white/70 px-1">checkpoints/hybrid/model_b_best.pth</code>.</li>
        <li>Open the <Link className="underline" to="/models">Model Laboratory</Link> and choose “Use for analysis”.</li>
      </ol>
    </Notice>
  );
}

export function ProbabilityMeter({ value, threshold, label }) {
  const v = Math.max(0, Math.min(1, value ?? 0));
  return (
    <div>
      <div className="flex items-baseline justify-between">
        <span className="text-xs text-navy-500">{label}</span>
        <span className="text-lg font-semibold tabular-nums text-navy">{fmt(value, 3)}</span>
      </div>
      <div className="relative mt-1.5 h-2.5 rounded-full bg-gradient-to-r from-slate-100 to-slate-200">
        <div className="absolute inset-y-0 left-0 rounded-full bg-brand/70" style={{ width: `${v * 100}%` }} />
        {threshold !== undefined && threshold !== null && (
          <div className="absolute -top-1 h-4.5 w-0.5 bg-navy" style={{ left: `${threshold * 100}%`, height: "18px" }}
            title={`Decision threshold ${fmt(threshold, 3)}`} />
        )}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-navy-300">
        <span>0</span>
        {threshold !== undefined && threshold !== null && <span>threshold {fmt(threshold, 2)}</span>}
        <span>1</span>
      </div>
    </div>
  );
}

export function ZoneBars({ zones, color = "#0e8f86" }) {
  if (!zones?.length) return null;
  return (
    <ul className="space-y-1.5">
      {zones.map((z) => (
        <li key={z.zone} className="text-[12px]">
          <div className="flex justify-between text-navy-500"><span className="first-letter:uppercase">{z.zone}</span><span>{(z.share * 100).toFixed(0)}%</span></div>
          <div className="mt-0.5 h-1.5 rounded-full bg-slate-100">
            <div className="h-full rounded-full" style={{ width: `${z.share * 100}%`, background: color }} />
          </div>
        </li>
      ))}
    </ul>
  );
}
