import { useEffect, useRef, useState } from "react";
import { FileImage, FileWarning, UploadCloud, X } from "lucide-react";
import { ACCEPTED, UPLOAD_LIMIT_MB } from "../services/api";

export function validateFile(file) {
  if (!file) return "No file selected.";
  const ext = "." + (file.name.split(".").pop() || "").toLowerCase();
  if (!ACCEPTED.includes(ext)) return `Unsupported file type (${ext}). Use PNG, JPEG or DICOM (.dcm).`;
  if (file.size === 0) return "The file is empty.";
  if (file.size > UPLOAD_LIMIT_MB * 1024 * 1024) return `File is larger than ${UPLOAD_LIMIT_MB} MB.`;
  return null;
}

const isDicom = (f) => /\.(dcm|dicom)$/i.test(f?.name || "");

export default function Dropzone({ file, onFile, disabled }) {
  const inputRef = useRef(null);
  const [drag, setDrag] = useState(false);
  const [error, setError] = useState(null);
  const [preview, setPreview] = useState(null);
  const [dims, setDims] = useState(null);

  useEffect(() => {
    setDims(null);
    if (!file || isDicom(file)) {
      setPreview(null);
      return undefined;
    }
    const url = URL.createObjectURL(file);
    setPreview(url);
    const img = new Image();
    img.onload = () => setDims({ w: img.naturalWidth, h: img.naturalHeight });
    img.onerror = () => setError("The browser could not display this image; it may be damaged.");
    img.src = url;
    return () => URL.revokeObjectURL(url);
  }, [file]);

  const accept = (f) => {
    const err = validateFile(f);
    setError(err);
    if (!err) onFile(f);
  };

  return (
    <div>
      {!file ? (
        <div
          onDragOver={(e) => { e.preventDefault(); if (!disabled) setDrag(true); }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            if (!disabled && e.dataTransfer.files?.[0]) accept(e.dataTransfer.files[0]);
          }}
          className={`flex flex-col items-center rounded-xl border-2 border-dashed px-6 py-10 text-center transition-colors ${
            drag ? "border-teal bg-teal-soft" : "border-line bg-canvas"
          } ${disabled ? "opacity-60" : ""}`}
        >
          <UploadCloud size={30} className="text-teal" />
          <p className="mt-3 text-sm font-medium text-navy">Drag and drop a chest X-ray here</p>
          <p className="mt-1 text-xs text-navy-500">PNG, JPEG or DICOM · frontal (PA/AP) view · up to {UPLOAD_LIMIT_MB} MB</p>
          <button
            type="button"
            disabled={disabled}
            onClick={() => inputRef.current?.click()}
            className="mt-4 rounded-lg bg-white px-4 py-2 text-sm font-medium text-navy ring-1 ring-line hover:bg-slate-50 disabled:cursor-not-allowed"
          >
            Browse files
          </button>
          <input
            ref={inputRef}
            type="file"
            accept={ACCEPTED.join(",")}
            className="hidden"
            onChange={(e) => { if (e.target.files?.[0]) accept(e.target.files[0]); e.target.value = ""; }}
          />
        </div>
      ) : (
        <div className="rounded-xl border border-line bg-canvas p-3">
          <div className="checker grid aspect-square w-full place-items-center overflow-hidden rounded-lg">
            {preview ? (
              <img src={preview} alt="Selected X-ray preview" className="max-h-full max-w-full object-contain" />
            ) : (
              <div className="flex flex-col items-center gap-2 p-6 text-center text-slate-300">
                <FileImage size={32} />
                <p className="text-xs">DICOM file selected. A preview appears after analysis.</p>
              </div>
            )}
          </div>
          <div className="mt-3 flex items-start justify-between gap-2">
            <div className="min-w-0 text-[13px]">
              <p className="truncate font-medium text-navy" title={file.name}>{file.name}</p>
              <p className="text-xs text-navy-500">
                {(file.size / 1024 / 1024).toFixed(2)} MB
                {dims ? ` · ${dims.w} × ${dims.h} px` : isDicom(file) ? " · DICOM" : ""}
              </p>
            </div>
            <button
              onClick={() => { onFile(null); setError(null); }}
              disabled={disabled}
              className="rounded p-1 text-navy-500 hover:bg-white"
              aria-label="Remove file"
            >
              <X size={16} />
            </button>
          </div>
        </div>
      )}
      {error && (
        <p className="mt-2 flex items-center gap-1.5 text-xs text-rose-700" role="alert">
          <FileWarning size={14} /> {error}
        </p>
      )}
    </div>
  );
}
