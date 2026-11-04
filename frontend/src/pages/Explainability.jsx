import { useState } from "react";
import { Brain, Eye, Info, Lightbulb, Ruler } from "lucide-react";
import { useApp } from "../context/AppContext";
import { api } from "../services/api";
import Dropzone from "../components/Dropzone";
import ImageViewer, { MapLegend } from "../components/ImageViewer";
import { NoTrainedModelNotice, ZoneBars } from "../components/common";
import {
  Badge, Button, Card, CardHeader, Collapsible, EmptyState, ErrorState, Notice, PageHeader, ProgressBar,
} from "../components/ui";
import { fmt, pct } from "../lib/format";

function parseBoxesText(text) {
  if (!text.trim()) return null;
  const rows = text.trim().split(/\n|;/).map((r) => r.trim()).filter(Boolean);
  const boxes = rows.map((r) => r.split(/[\s,]+/).map(Number));
  if (boxes.some((b) => b.length !== 4 || b.some((v) => !Number.isFinite(v)) || b[2] <= 0 || b[3] <= 0))
    throw new Error("Each line must contain four numbers: x, y, width, height (in original image pixels).");
  return JSON.stringify(boxes);
}

function Panel({ title, subtitle, children }) {
  return (
    <Card className="min-w-0">
      <div className="border-b border-line px-4 py-3">
        <p className="text-sm font-semibold text-navy">{title}</p>
        <p className="text-xs text-navy-500">{subtitle}</p>
      </div>
      <div className="p-3">{children}</div>
    </Card>
  );
}

const hit = (v) => (v === true ? <Badge tone="good">hit</Badge> : v === false ? <Badge tone="bad">miss</Badge> : "—");

function MetricsTable({ metrics }) {
  const rows = [
    ["Predicted localization", metrics.localization],
    ["Grad-CAM++", metrics.gradcam],
  ].filter(([, m]) => m);
  return (
    <div className="overflow-x-auto">
      <table className="w-full min-w-[520px] text-[13px]">
        <thead className="text-left text-xs text-navy-500">
          <tr><th className="py-2">Map</th><th>IoU</th><th>Dice</th><th>Pointing game</th><th>Share inside box</th></tr>
        </thead>
        <tbody className="divide-y divide-line">
          {rows.map(([name, m]) => (
            <tr key={name}>
              <td className="py-2 font-medium">{name}</td>
              <td>{fmt(m.iou)}</td><td>{fmt(m.dice)}</td><td>{hit(m.pointing_game_hit)}</td><td>{pct(m.mass_in_box)}</td>
            </tr>
          ))}
          {metrics.attention_alignment && (
            <tr>
              <td className="py-2 font-medium">Spatial attention</td><td>—</td><td>—</td>
              <td>{metrics.attention_alignment.pointing_game === 1 ? hit(true) : metrics.attention_alignment.pointing_game === 0 ? hit(false) : "—"}</td>
              <td>{pct(metrics.attention_alignment.mass_in_box)}</td>
            </tr>
          )}
        </tbody>
      </table>
      <p className="mt-2 text-[11px] text-navy-300">
        IoU/Dice use a 0.5 threshold. For Grad-CAM++ the map is rescaled to 0–1 first, so its IoU/Dice depend on an arbitrary cut-off; the pointing game and “share inside box” do not need a threshold.
      </p>
    </div>
  );
}

function influenceSentence(zones, what) {
  if (!zones?.length) return null;
  const top = zones[0];
  return `${what} is strongest in the ${top.zone} zone (${(top.share * 100).toFixed(0)}% of the total), followed by the ${zones[1]?.zone} zone (${((zones[1]?.share || 0) * 100).toFixed(0)}%).`;
}

export default function Explainability() {
  const { trainedModels, selectedModel, models, lastFile, setLastFile, apiReachable } = useApp();
  const [file, setFile] = useState(lastFile);
  const [boxesText, setBoxesText] = useState("");
  const [busy, setBusy] = useState(false);
  const [progress, setProgress] = useState(null);
  const [error, setError] = useState(null);
  const [res, setRes] = useState(null);
  const noModels = models && !trainedModels.length;

  const run = async () => {
    setError(null);
    let boxes = null;
    try {
      boxes = parseBoxesText(boxesText);
    } catch (e) {
      setError(e);
      return;
    }
    setBusy(true);
    try {
      setRes(await api.explain(file, selectedModel, boxes, setProgress));
    } catch (e) {
      setError(e);
    } finally {
      setBusy(false);
      setProgress(null);
    }
  };

  const ref = res?.reference;
  return (
    <div>
      <PageHeader title="Explainability" subtitle="Compare where the model predicts the opacity with what influenced its score, and - when a radiologist annotation exists - how well they agree." />
      {noModels && <div className="mb-6"><NoTrainedModelNotice /></div>}

      <Card className="mb-6">
        <CardHeader icon={Eye} title="Image and optional reference annotation" />
        <div className="grid gap-6 p-5 md:grid-cols-[320px_minmax(0,1fr)]">
          <Dropzone file={file} onFile={(f) => { setFile(f); setLastFile(f); }} disabled={busy} />
          <div className="space-y-3">
            <label className="block text-sm font-medium text-navy" htmlFor="boxes">Radiologist boxes (optional)</label>
            <textarea id="boxes" rows={4} value={boxesText} onChange={(e) => setBoxesText(e.target.value)}
              placeholder={"x, y, width, height  (one box per line, original image pixels)\n264, 152, 213, 379"}
              className="w-full rounded-lg border border-line px-3 py-2 font-mono text-xs text-navy placeholder:text-navy-300" />
            <p className="text-xs text-navy-500">
              If left empty and the file is an RSNA image named <span className="font-mono">&lt;patientId&gt;.dcm</span>, the matching RSNA annotation is loaded automatically from the local dataset.
            </p>
            <Button icon={Brain} onClick={run} loading={busy} disabled={!file || !selectedModel || !apiReachable}>Generate explanations</Button>
            {progress && <ProgressBar value={progress.phase === "upload" ? progress.fraction : 1} label={progress.phase === "upload" ? "Uploading…" : "Computing maps…"} />}
            <ErrorState error={error} compact />
          </div>
        </div>
      </Card>

      {!res && !busy && (
        <Card><EmptyState icon={Eye} title="No explanation yet">Upload an image and press “Generate explanations”.</EmptyState></Card>
      )}

      {res && (
        <div className="space-y-6">
          <div className="grid gap-4 md:grid-cols-3">
            <Panel title="Original X-ray" subtitle={ref.available ? `Green dashed boxes: reference annotation (${ref.source === "provided" ? "entered by you" : "RSNA dataset"})` : "No reference annotation available"}>
              <ImageViewer base={res.images.original} boxes={ref.boxes} size={res.image_size} />
            </Panel>
            <Panel title="Predicted localization" subtitle="The model's own prediction of the opacity location">
              {res.images.localization ? (
                <ImageViewer base={res.images.original} layers={res.images} visible={["localization"]} boxes={ref.boxes} size={res.image_size} />
              ) : (
                <EmptyState title="Not available">{res.localization.reason}</EmptyState>
              )}
            </Panel>
            <Panel title="Grad-CAM++" subtitle="Post-hoc explanation of the classification score">
              <ImageViewer base={res.images.original} layers={res.images} visible={["gradcam"]} boxes={ref.boxes} size={res.image_size} />
            </Panel>
          </div>

          <div className="grid gap-6 lg:grid-cols-2">
            <Card>
              <CardHeader icon={Lightbulb} title="What influenced this model's prediction?" subtitle={`${res.model.name} · ${res.prediction.label} (score ${fmt(res.prediction.probability, 3)})`} />
              <div className="space-y-4 p-5 text-[13px] text-navy-700">
                <p>{influenceSentence(res.influence.gradcam_zones, "Grad-CAM++ influence")}</p>
                <ZoneBars zones={res.influence.gradcam_zones.slice(0, 4)} color="#d9480f" />
                {res.influence.localization_zones && (
                  <>
                    <p>{influenceSentence(res.influence.localization_zones, "The predicted localization")}</p>
                    <ZoneBars zones={res.influence.localization_zones.slice(0, 4)} />
                  </>
                )}
                <p className="text-xs text-navy-500">{res.orientation_note}</p>
                <Notice tone="warn" icon={Info}>These are computational outputs of the model, not proof of clinical reasoning.</Notice>
              </div>
            </Card>
            <Card>
              <CardHeader icon={Ruler} title="Agreement with the reference annotation" />
              <div className="p-5">
                {ref.available ? <MetricsTable metrics={res.metrics} /> : (
                  <EmptyState icon={Ruler} title="No reference annotation">
                    IoU, Dice and the pointing game need a radiologist box. Enter one above, or use an RSNA image from the local dataset.
                  </EmptyState>
                )}
              </div>
            </Card>
          </div>

          {res.images.attention && (
            <Collapsible title="Researcher view: spatial attention of the hybrid classifier">
              <div className="grid gap-4 md:grid-cols-[minmax(0,360px)_1fr]">
                <ImageViewer base={res.images.original} layers={res.images} visible={["attention"]} boxes={ref.boxes} size={res.image_size} />
                <div className="space-y-2">
                  <p>{res.map_descriptions.attention}</p>
                  <MapLegend kind="attention" />
                  {res.metrics.attention_alignment && (
                    <p>Attention mass inside the reference box: <b>{pct(res.metrics.attention_alignment.mass_in_box)}</b> · KL divergence to box: {fmt(res.metrics.attention_alignment.kl_to_box)}</p>
                  )}
                </div>
              </div>
            </Collapsible>
          )}

          <Card>
            <CardHeader icon={Info} title="Limitations of these explanations" />
            <ul className="list-disc space-y-1.5 px-9 py-4 text-[13px] text-navy-700">
              {res.limitations.map((l) => <li key={l}>{l}</li>)}
            </ul>
          </Card>
        </div>
      )}
    </div>
  );
}
