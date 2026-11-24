import { useHealth } from '../health'

const STYLES = {
  model: { dot: 'bg-emerald-500', text: 'text-emerald-800', bg: 'bg-emerald-50 border-emerald-200' },
  demo: { dot: 'bg-amber-500', text: 'text-amber-900', bg: 'bg-amber-50 border-amber-200' },
  loading: { dot: 'bg-sky-500 animate-pulse', text: 'text-sky-900', bg: 'bg-sky-50 border-sky-200' },
  offline: { dot: 'bg-rose-500', text: 'text-rose-900', bg: 'bg-rose-50 border-rose-200' },
  checking: { dot: 'bg-slate-400 animate-pulse', text: 'text-slate-700', bg: 'bg-slate-50 border-slate-200' },
}

export function statusOf(health, offline) {
  if (offline) return { key: 'offline', label: 'API offline' }
  if (!health) return { key: 'checking', label: 'Connecting…' }
  if (health.mode === 'model') return { key: 'model', label: `Model ready · ${health.device.toUpperCase()}` }
  if (health.mode === 'loading') return { key: 'loading', label: 'Loading model…' }
  return { key: 'demo', label: 'Demo mode' }
}

export default function StatusPill({ className = '' }) {
  const { health, offline } = useHealth()
  const s = statusOf(health, offline)
  const c = STYLES[s.key]
  return (
    <span
      title={offline ? 'Start the API: python server/app.py' : health?.reason}
      className={`inline-flex items-center gap-2 rounded-full border px-3 py-1 text-xs font-semibold ${c.bg} ${c.text} ${className}`}
    >
      <span className={`h-2 w-2 rounded-full ${c.dot}`} />
      {s.label}
    </span>
  )
}
