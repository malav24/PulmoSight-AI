import { useState } from "react";
import { AlertTriangle, ChevronDown, Inbox, Loader2, RefreshCw } from "lucide-react";

export function Card({ className = "", children, ...rest }) {
  return (
    <section className={`rounded-xl2 border border-line bg-white shadow-card ${className}`} {...rest}>
      {children}
    </section>
  );
}

export function CardHeader({ icon: Icon, title, subtitle, action }) {
  return (
    <div className="flex items-start justify-between gap-3 border-b border-line px-5 py-4">
      <div className="flex min-w-0 items-start gap-3">
        {Icon && (
          <div className="mt-0.5 rounded-lg bg-brand-soft p-2 text-brand">
            <Icon size={18} />
          </div>
        )}
        <div className="min-w-0">
          <h2 className="text-[15px] font-semibold text-navy">{title}</h2>
          {subtitle && <p className="mt-0.5 text-[13px] text-navy-500">{subtitle}</p>}
        </div>
      </div>
      {action}
    </div>
  );
}

const TONES = {
  good: "bg-teal-soft text-teal-600 ring-teal/20",
  warn: "bg-amber-50 text-amber-700 ring-amber-200",
  bad: "bg-rose-50 text-rose-700 ring-rose-200",
  info: "bg-brand-soft text-brand-600 ring-brand/20",
  neutral: "bg-slate-100 text-slate-600 ring-slate-200",
};

export function Badge({ tone = "neutral", children, icon: Icon }) {
  return (
    <span className={`inline-flex items-center gap-1 whitespace-nowrap rounded-full px-2.5 py-0.5 text-xs font-medium ring-1 ${TONES[tone]}`}>
      {Icon && <Icon size={12} />}
      {children}
    </span>
  );
}

export function Button({ variant = "primary", icon: Icon, loading, className = "", children, ...rest }) {
  const styles = {
    primary: "bg-teal text-white hover:bg-teal-600 disabled:bg-slate-300",
    secondary: "bg-white text-navy ring-1 ring-line hover:bg-slate-50 disabled:text-slate-400",
    ghost: "text-navy-500 hover:bg-slate-100 disabled:text-slate-300",
    blue: "bg-brand text-white hover:bg-brand-600 disabled:bg-slate-300",
  };
  return (
    <button
      className={`inline-flex items-center justify-center gap-2 rounded-lg px-4 py-2 text-sm font-medium transition-colors disabled:cursor-not-allowed ${styles[variant]} ${className}`}
      disabled={loading || rest.disabled}
      {...rest}
    >
      {loading ? <Loader2 size={16} className="animate-spin" /> : Icon && <Icon size={16} />}
      {children}
    </button>
  );
}

export function Spinner({ label = "Loading…" }) {
  return (
    <div className="flex items-center gap-2 p-6 text-sm text-navy-500" role="status">
      <Loader2 size={18} className="animate-spin text-teal" /> {label}
    </div>
  );
}

export function EmptyState({ icon: Icon = Inbox, title, children, action }) {
  return (
    <div className="flex flex-col items-center px-6 py-10 text-center">
      <div className="rounded-full bg-slate-100 p-3 text-navy-300">
        <Icon size={22} />
      </div>
      <p className="mt-3 text-sm font-semibold text-navy">{title}</p>
      {children && <div className="mt-1 max-w-md text-[13px] text-navy-500">{children}</div>}
      {action && <div className="mt-4">{action}</div>}
    </div>
  );
}

export function ErrorState({ error, onRetry, compact }) {
  if (!error) return null;
  return (
    <div className={`flex items-start gap-3 rounded-lg border border-rose-200 bg-rose-50 text-rose-800 ${compact ? "p-3" : "p-4"}`} role="alert">
      <AlertTriangle size={18} className="mt-0.5 shrink-0" />
      <div className="min-w-0 flex-1 text-sm">
        <p className="font-medium">{error.message || String(error)}</p>
        {error.code && error.code !== "http_error" && <p className="mt-0.5 text-xs opacity-70">Code: {error.code}</p>}
      </div>
      {onRetry && (
        <button onClick={onRetry} className="inline-flex items-center gap-1 text-xs font-medium hover:underline">
          <RefreshCw size={12} /> Retry
        </button>
      )}
    </div>
  );
}

export function Notice({ tone = "info", icon: Icon, title, children }) {
  const s = {
    info: "border-brand/20 bg-brand-soft text-navy-700",
    warn: "border-amber-200 bg-amber-50 text-amber-900",
    teal: "border-teal/20 bg-teal-soft text-navy-700",
  }[tone];
  return (
    <div className={`flex gap-3 rounded-lg border p-4 text-sm ${s}`}>
      {Icon && <Icon size={18} className="mt-0.5 shrink-0" />}
      <div className="min-w-0">
        {title && <p className="font-semibold">{title}</p>}
        <div className={title ? "mt-1" : ""}>{children}</div>
      </div>
    </div>
  );
}

export function Collapsible({ title = "Technical details", children, defaultOpen = false }) {
  const [open, setOpen] = useState(defaultOpen);
  return (
    <div className="rounded-lg border border-line">
      <button
        onClick={() => setOpen((o) => !o)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-left text-[13px] font-medium text-navy-500 hover:bg-slate-50"
        aria-expanded={open}
      >
        {title}
        <ChevronDown size={16} className={`transition-transform ${open ? "rotate-180" : ""}`} />
      </button>
      {open && <div className="border-t border-line px-4 py-3 text-[13px] text-navy-700">{children}</div>}
    </div>
  );
}

export function Stat({ label, value, hint, tone }) {
  return (
    <div className="rounded-lg bg-canvas px-4 py-3">
      <p className="text-xs font-medium uppercase tracking-wide text-navy-500">{label}</p>
      <p className={`mt-1 text-xl font-semibold ${tone === "muted" ? "text-navy-300" : "text-navy"}`}>{value}</p>
      {hint && <p className="mt-0.5 text-xs text-navy-500">{hint}</p>}
    </div>
  );
}

export function Tabs({ tabs, value, onChange }) {
  return (
    <div className="flex flex-wrap gap-1 rounded-lg bg-canvas p-1" role="tablist">
      {tabs.map((t) => (
        <button
          key={t.value}
          role="tab"
          aria-selected={value === t.value}
          disabled={t.disabled}
          onClick={() => onChange(t.value)}
          className={`rounded-md px-3 py-1.5 text-[13px] font-medium transition-colors disabled:cursor-not-allowed disabled:text-slate-300 ${
            value === t.value ? "bg-white text-navy shadow-sm" : "text-navy-500 hover:text-navy"
          }`}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

export function KeyValue({ rows }) {
  return (
    <dl className="divide-y divide-line text-[13px]">
      {rows.filter(Boolean).map(([k, v]) => (
        <div key={k} className="flex justify-between gap-4 py-2">
          <dt className="text-navy-500">{k}</dt>
          <dd className="text-right font-medium text-navy">{v}</dd>
        </div>
      ))}
    </dl>
  );
}

export function PageHeader({ title, subtitle, actions }) {
  return (
    <div className="mb-6 flex flex-col gap-3 sm:flex-row sm:items-end sm:justify-between">
      <div>
        <h1 className="text-2xl font-semibold tracking-tight text-navy">{title}</h1>
        {subtitle && <p className="mt-1 max-w-3xl text-sm text-navy-500">{subtitle}</p>}
      </div>
      {actions && <div className="flex flex-wrap gap-2">{actions}</div>}
    </div>
  );
}

export function ProgressBar({ value, label }) {
  return (
    <div>
      <div className="h-1.5 w-full overflow-hidden rounded-full bg-slate-100">
        <div className="h-full rounded-full bg-teal transition-all" style={{ width: `${Math.round((value || 0) * 100)}%` }} />
      </div>
      {label && <p className="mt-1 text-xs text-navy-500">{label}</p>}
    </div>
  );
}
