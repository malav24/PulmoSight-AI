import { useRef, useState } from "react";
import { Columns2, Maximize, ZoomIn, ZoomOut } from "lucide-react";
import { MAP_INFO } from "../lib/format";

export function MapLegend({ kind }) {
  const info = MAP_INFO[kind];
  if (!info) return null;
  return (
    <div className="text-[11px] text-navy-500">
      <div className="h-2 w-40 rounded-full" style={{ background: info.gradient }} />
      <p className="mt-1">{info.title}: {info.scale}</p>
    </div>
  );
}

/**
 * Layered X-ray viewer.
 *  base:    data URL of the preprocessed X-ray
 *  layers:  { localization, gradcam, attention } RGBA data URLs (semi-transparent heatmaps)
 *  visible: array of layer names to show
 *  boxes:   reference boxes [[x1,y1,x2,y2]] in image pixels; size: image side in pixels
 */
export default function ImageViewer({ base, layers = {}, visible = [], boxes = [], size = 384, label, compareDefault = false }) {
  const [zoom, setZoom] = useState(1);
  const [pan, setPan] = useState({ x: 0, y: 0 });
  const [compare, setCompare] = useState(compareDefault);
  const [split, setSplit] = useState(50);
  const [opacity, setOpacity] = useState({ localization: 0.85, gradcam: 0.8, attention: 0.8 });
  const drag = useRef(null);
  const shown = visible.filter((v) => layers[v]);

  const clampZoom = (z) => Math.min(6, Math.max(1, z));
  const onWheel = (e) => {
    if (!e.ctrlKey && !e.metaKey && zoom === 1 && e.deltaY > 0) return;
    e.preventDefault();
    setZoom((z) => clampZoom(z * (e.deltaY < 0 ? 1.15 : 1 / 1.15)));
  };
  const reset = () => { setZoom(1); setPan({ x: 0, y: 0 }); };

  return (
    <div>
      <div
        className="checker relative aspect-square w-full touch-none select-none overflow-hidden rounded-lg"
        onWheel={onWheel}
        onPointerDown={(e) => { if (zoom > 1) drag.current = { x: e.clientX - pan.x, y: e.clientY - pan.y }; }}
        onPointerMove={(e) => { if (drag.current) setPan({ x: e.clientX - drag.current.x, y: e.clientY - drag.current.y }); }}
        onPointerUp={() => (drag.current = null)}
        onPointerLeave={() => (drag.current = null)}
        style={{ cursor: zoom > 1 ? "grab" : "default" }}
      >
        <div
          className="absolute inset-0 origin-center"
          style={{ transform: `translate(${pan.x}px, ${pan.y}px) scale(${zoom})` }}
        >
          {base && <img src={base} alt={label || "Chest X-ray"} className="absolute inset-0 h-full w-full" draggable={false} />}
          <div className="absolute inset-0" style={compare ? { clipPath: `inset(0 0 0 ${split}%)` } : undefined}>
            {shown.map((name) => (
              <img
                key={name}
                src={layers[name]}
                alt={`${MAP_INFO[name]?.title} overlay`}
                className="absolute inset-0 h-full w-full"
                style={{ opacity: opacity[name] }}
                draggable={false}
              />
            ))}
          </div>
          {boxes?.length > 0 && (
            <svg viewBox={`0 0 ${size} ${size}`} className="absolute inset-0 h-full w-full" aria-label="Reference annotation boxes">
              {boxes.map((b, i) => (
                <rect key={i} x={b[0]} y={b[1]} width={b[2] - b[0]} height={b[3] - b[1]} fill="none"
                  stroke="#34d399" strokeWidth={size / 192} strokeDasharray={`${size / 64} ${size / 128}`} />
              ))}
            </svg>
          )}
        </div>
        {compare && shown.length > 0 && (
          <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-white/90" style={{ left: `${split}%` }}>
            <span className="absolute left-1 top-2 rounded bg-black/50 px-1.5 py-0.5 text-[10px] text-white">overlay →</span>
            <span className="absolute right-1 top-2 rounded bg-black/50 px-1.5 py-0.5 text-[10px] text-white">← original</span>
          </div>
        )}
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <div className="flex items-center rounded-lg ring-1 ring-line">
          <button className="p-2 text-navy-500 hover:text-navy" onClick={() => setZoom((z) => clampZoom(z * 1.25))} aria-label="Zoom in"><ZoomIn size={16} /></button>
          <span className="w-12 text-center text-xs tabular-nums text-navy-500">{Math.round(zoom * 100)}%</span>
          <button className="p-2 text-navy-500 hover:text-navy" onClick={() => setZoom((z) => clampZoom(z / 1.25))} aria-label="Zoom out"><ZoomOut size={16} /></button>
          <button className="border-l border-line p-2 text-navy-500 hover:text-navy" onClick={reset} aria-label="Reset view"><Maximize size={16} /></button>
        </div>
        {shown.length > 0 && (
          <button
            onClick={() => setCompare((c) => !c)}
            className={`inline-flex items-center gap-1.5 rounded-lg px-3 py-2 text-xs font-medium ring-1 ${compare ? "bg-teal-soft text-teal-600 ring-teal/30" : "text-navy-500 ring-line hover:text-navy"}`}
          >
            <Columns2 size={14} /> Compare with original
          </button>
        )}
        {compare && shown.length > 0 && (
          <input type="range" min={0} max={100} value={split} onChange={(e) => setSplit(+e.target.value)}
            className="w-32" aria-label="Comparison divider position" />
        )}
      </div>
      {shown.length > 0 && (
        <div className="mt-3 grid gap-3 sm:grid-cols-2">
          {shown.map((name) => (
            <div key={name} className="rounded-lg bg-canvas p-3">
              <MapLegend kind={name} />
              <label className="mt-2 flex items-center gap-2 text-[11px] text-navy-500">
                Opacity
                <input type="range" min={0} max={1} step={0.05} value={opacity[name]}
                  onChange={(e) => setOpacity((o) => ({ ...o, [name]: +e.target.value }))} className="flex-1"
                  aria-label={`${MAP_INFO[name]?.title} opacity`} />
              </label>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
