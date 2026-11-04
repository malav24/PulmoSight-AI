import { useEffect, useState } from "react";
import { AlertTriangle, CheckCircle2, Database, FolderOpen, RefreshCw, XCircle } from "lucide-react";
import { api } from "../services/api";
import { Badge, Button, Card, CardHeader, Collapsible, ErrorState, Notice, PageHeader, Spinner } from "../components/ui";
import { pct } from "../lib/format";

function Check({ ok, label, detail }) {
  return (
    <li className="flex items-start justify-between gap-3 py-2 text-[13px]">
      <span className="flex items-center gap-2">
        {ok ? <CheckCircle2 size={16} className="text-teal" /> : <XCircle size={16} className="text-slate-300" />}
        {label}
      </span>
      {detail !== undefined && <span className="text-right text-navy-500">{detail}</span>}
    </li>
  );
}

function Steps({ items }) {
  return (
    <ol className="list-decimal space-y-1.5 pl-5 text-[13px] text-navy-700">
      {items.map((s) => <li key={s}>{s}</li>)}
    </ol>
  );
}

export default function Datasets() {
  const [data, setData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const load = async (refresh = false) => {
    setLoading(true);
    try {
      setData(await api.datasets(refresh));
      setError(null);
    } catch (e) {
      setError(e);
    } finally {
      setLoading(false);
    }
  };
  useEffect(() => { load(); }, []);

  if (error) return <ErrorState error={error} onRetry={() => load(true)} />;
  if (!data) return <Spinner label="Checking dataset folders…" />;
  const { rsna, nih } = data;
  const splits = rsna.split_counts && typeof rsna.split_counts === "object" ? rsna.split_counts : null;

  return (
    <div className="space-y-6">
      <PageHeader title="Dataset Management" subtitle="What the server actually finds on disk. Nothing here is estimated - counts come from the files present."
        actions={<Button variant="secondary" icon={RefreshCw} loading={loading} onClick={() => load(true)}>Re-scan</Button>} />
      <div className="grid gap-6 lg:grid-cols-2">
        <Card>
          <CardHeader icon={Database} title={rsna.name} subtitle={rsna.role}
            action={<Badge tone={rsna.split_file ? "good" : rsna.ready_for_training ? "info" : "neutral"}>{rsna.preparation_status}</Badge>} />
          <div className="p-5">
            <p className="mb-2 flex items-center gap-2 text-xs text-navy-500"><FolderOpen size={14} /> Folder: <code>{rsna.directory}</code></p>
            <ul className="divide-y divide-line">
              <Check ok={rsna.directory_exists} label="Dataset folder present" />
              <Check ok={rsna.annotation_csv} label="Annotation CSV (stage_2_train_labels.csv)" />
              <Check ok={rsna.class_info_csv} label="Class information CSV" />
              <Check ok={rsna.images_available} label="DICOM images" detail={`${rsna.image_count.toLocaleString()} found`} />
              <Check ok={rsna.split_file} label="Patient-level train/val/test split" />
              <Check ok={rsna.cache_ready} label="Preprocessed image cache" detail={rsna.cache_configured ? undefined : "optional"} />
            </ul>
            {splits && (
              <table className="mt-4 w-full text-[13px]">
                <thead className="text-left text-xs text-navy-500"><tr><th className="pb-1">Split</th><th>Images</th><th>With opacity</th><th>Share</th></tr></thead>
                <tbody className="divide-y divide-line">
                  {["train", "val", "test"].filter((s) => splits[s]).map((s) => (
                    <tr key={s}><td className="py-1.5 capitalize">{s}</td><td>{splits[s].images.toLocaleString()}</td><td>{splits[s].positives.toLocaleString()}</td><td>{pct(splits[s].positives / splits[s].images)}</td></tr>
                  ))}
                </tbody>
              </table>
            )}
            <div className="mt-4"><Collapsible title="How to add the RSNA dataset" defaultOpen={!rsna.ready_for_training}><Steps items={data.instructions.rsna} /></Collapsible></div>
          </div>
        </Card>

        <Card>
          <CardHeader icon={Database} title={nih.name} subtitle={nih.role}
            action={<Badge tone={nih.ready_for_external_evaluation ? "good" : "neutral"}>{nih.ready_for_external_evaluation ? "ready" : "not ready"}</Badge>} />
          <div className="p-5">
            <p className="mb-2 flex items-center gap-2 text-xs text-navy-500"><FolderOpen size={14} /> Folder: <code>{nih.directory}</code>
              {nih.variant && <Badge tone={nih.variant.startsWith("random") ? "warn" : "info"}>{nih.variant}</Badge>}</p>
            {nih.variant?.startsWith("random") && (
              <p className="mb-2 text-xs text-amber-800">Only the ~5 % NIH random sample is installed: few Pneumonia-labelled images, so external results will have wide confidence intervals.</p>
            )}
            <ul className="divide-y divide-line">
              <Check ok={nih.directory_exists} label="Dataset folder present" />
              <Check ok={nih.metadata_csv} label="Metadata CSV (Data_Entry_2017 or sample_labels)" />
              <Check ok={nih.test_list} label="Official test list" detail={nih.variant?.startsWith("random") ? "not in sample" : undefined} />
              <Check ok={nih.bbox_csv} label="Official bounding-box list" detail="optional" />
              <Check ok={nih.rsna_overlap_mapping} label="RSNA overlap mapping file" />
              <Check ok={nih.images_available} label="PNG images" detail={`${nih.image_count.toLocaleString()} found`} />
            </ul>
            {nih.external_validity_warning && nih.metadata_csv && (
              <div className="mt-4"><Notice tone="warn" icon={AlertTriangle}>{nih.external_validity_warning}</Notice></div>
            )}
            {nih.statistics?.external_set && (
              <p className="mt-3 text-[13px] text-navy-700">Prepared external set: {nih.statistics.external_set.n_images.toLocaleString()} images, {nih.statistics.external_set.n_pneumonia.toLocaleString()} labelled Pneumonia.</p>
            )}
            <div className="mt-4"><Collapsible title="How to add the NIH dataset" defaultOpen={!nih.ready_for_external_evaluation}><Steps items={data.instructions.nih} /></Collapsible></div>
          </div>
        </Card>
      </div>
      {rsna.statistics && (
        <Card className="p-5 text-[13px]">
          <p className="font-semibold">RSNA statistics (from scripts/prepare_rsna.py)</p>
          <p className="mt-1 text-navy-700">{rsna.statistics.n_images?.toLocaleString()} images · {rsna.statistics.n_positive?.toLocaleString()} with annotated opacity ({pct(rsna.statistics.prevalence)}) · {rsna.statistics.n_boxes?.toLocaleString()} boxes</p>
        </Card>
      )}
      <p className="text-xs text-navy-500">Last checked {data.checked_at.replace("T", " ")}. Dataset images are never uploaded, displayed or shared by this application.</p>
    </div>
  );
}
