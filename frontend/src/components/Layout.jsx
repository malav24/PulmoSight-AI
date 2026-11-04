import { useState } from "react";
import { NavLink, useLocation } from "react-router-dom";
import {
  BarChart3, BookOpen, Cpu, Database, Eye, FlaskConical, LayoutDashboard, Menu, ScanLine, ShieldAlert, X,
} from "lucide-react";
import { useApp } from "../context/AppContext";

export const NAV = [
  { to: "/", label: "Overview", icon: LayoutDashboard },
  { to: "/analysis", label: "X-ray Analysis", icon: ScanLine },
  { to: "/explainability", label: "Explainability", icon: Eye },
  { to: "/models", label: "Model Laboratory", icon: Cpu },
  { to: "/experiments", label: "Research Experiments", icon: BarChart3 },
  { to: "/datasets", label: "Dataset Management", icon: Database },
  { to: "/about", label: "About Research", icon: BookOpen },
];

export const DISCLAIMER =
  "For research and educational purposes only. This system is not a medical device and must not be used as a substitute for professional radiological interpretation or clinical diagnosis.";

function ApiStatus() {
  const { health, healthError } = useApp();
  if (healthError)
    return (
      <span className="flex items-center gap-1.5 text-xs text-rose-700">
        <span className="h-2 w-2 rounded-full bg-rose-500" /> Server unreachable
      </span>
    );
  if (!health)
    return (
      <span className="flex items-center gap-1.5 text-xs text-navy-500">
        <span className="h-2 w-2 rounded-full bg-slate-300" /> Connecting…
      </span>
    );
  return (
    <span className="flex items-center gap-1.5 text-xs text-navy-500">
      <span className={`h-2 w-2 rounded-full ${health.status === "ok" ? "bg-teal" : "bg-amber-500"}`} />
      Server online · {health.ml?.device?.toUpperCase?.() || "—"}
    </span>
  );
}

function ModelPicker() {
  const { trainedModels, selectedModel, setSelectedModel, models } = useApp();
  if (!models) return null;
  if (!trainedModels.length)
    return <span className="rounded-full bg-slate-100 px-3 py-1 text-xs font-medium text-navy-500">Models not trained yet</span>;
  return (
    <label className="flex items-center gap-2 text-xs text-navy-500">
      <span className="hidden sm:inline">Model</span>
      <select
        value={selectedModel || ""}
        onChange={(e) => setSelectedModel(e.target.value)}
        className="rounded-lg border border-line bg-white px-2 py-1.5 text-[13px] font-medium text-navy"
        aria-label="Model used for analysis"
      >
        {trainedModels.map((m) => (
          <option key={m.key} value={m.key}>{m.name}</option>
        ))}
      </select>
    </label>
  );
}

function Sidebar({ onNavigate }) {
  return (
    <nav className="flex h-full flex-col">
      <div className="px-5 pb-6 pt-6">
        <div className="flex items-center gap-2.5">
          <div className="grid h-9 w-9 place-items-center rounded-lg bg-navy text-teal-soft">
            <FlaskConical size={18} />
          </div>
          <div className="leading-tight">
            <p className="text-[13px] font-bold tracking-wide text-navy">RIGHT FOR THE</p>
            <p className="text-[13px] font-bold tracking-wide text-teal">RIGHT REASON</p>
          </div>
        </div>
        <p className="mt-3 text-xs leading-snug text-navy-500">Explanation-Guided Pneumonia Detection and Localization</p>
      </div>
      <ul className="flex-1 space-y-0.5 px-3">
        {NAV.map(({ to, label, icon: Icon }) => (
          <li key={to}>
            <NavLink
              to={to}
              end={to === "/"}
              onClick={onNavigate}
              className={({ isActive }) =>
                `flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium transition-colors ${
                  isActive ? "bg-brand-soft text-brand-600" : "text-navy-500 hover:bg-slate-50 hover:text-navy"
                }`
              }
            >
              <Icon size={18} /> {label}
            </NavLink>
          </li>
        ))}
      </ul>
      <div className="m-3 rounded-lg bg-canvas p-3 text-[11px] leading-snug text-navy-500">
        Research prototype · not a medical device
      </div>
    </nav>
  );
}

export default function Layout({ children }) {
  const [open, setOpen] = useState(false);
  const { pathname } = useLocation();
  const current = NAV.find((n) => (n.to === "/" ? pathname === "/" : pathname.startsWith(n.to)));
  return (
    <div className="flex min-h-full">
      <aside className="sticky top-0 hidden h-screen w-64 shrink-0 border-r border-line bg-white lg:block">
        <Sidebar />
      </aside>
      {open && (
        <div className="fixed inset-0 z-40 lg:hidden" role="dialog" aria-modal="true">
          <div className="absolute inset-0 bg-navy/30" onClick={() => setOpen(false)} />
          <aside className="absolute inset-y-0 left-0 w-72 bg-white shadow-xl">
            <button className="absolute right-3 top-3 rounded p-1 text-navy-500" onClick={() => setOpen(false)} aria-label="Close menu">
              <X size={20} />
            </button>
            <Sidebar onNavigate={() => setOpen(false)} />
          </aside>
        </div>
      )}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="sticky top-0 z-30 flex items-center gap-3 border-b border-line bg-white/95 px-4 py-3 backdrop-blur sm:px-6">
          <button className="rounded p-1 text-navy lg:hidden" onClick={() => setOpen(true)} aria-label="Open menu">
            <Menu size={22} />
          </button>
          <p className="min-w-0 flex-1 truncate text-sm font-semibold text-navy">{current?.label}</p>
          <ApiStatus />
          <ModelPicker />
        </header>
        <div className="border-b border-amber-200 bg-amber-50 px-4 py-2 sm:px-6">
          <p className="flex items-start gap-2 text-[12.5px] leading-snug text-amber-900">
            <ShieldAlert size={15} className="mt-0.5 shrink-0" />
            <span>{DISCLAIMER}</span>
          </p>
        </div>
        <main className="mx-auto w-full max-w-7xl flex-1 px-4 py-6 sm:px-6 lg:py-8">{children}</main>
      </div>
    </div>
  );
}
