import { useEffect, useState } from 'react'
import { NavLink, Outlet, useLocation, Link } from 'react-router-dom'
import { Menu, X, Github } from 'lucide-react'
import StatusPill from './StatusPill'

const LINKS = [
  { to: '/', label: 'Overview', end: true },
  { to: '/studio', label: 'Studio' },
  { to: '/architecture', label: 'Architecture' },
  { to: '/gallery', label: 'Gallery' },
]

export const REPO_URL = 'https://github.com/choudharyyuvraj/SYNGEN'

export function Logo() {
  return (
    <Link to="/" className="flex items-center gap-2.5">
      <img src="/favicon.svg" alt="" className="h-8 w-8" />
      <span className="font-display text-xl font-semibold tracking-tight text-ink">
        SYN<span className="font-hand text-2xl font-bold text-accent">gen</span>
      </span>
    </Link>
  )
}

export default function Layout() {
  const [open, setOpen] = useState(false)
  const { pathname } = useLocation()
  useEffect(() => { setOpen(false); window.scrollTo(0, 0) }, [pathname])

  const linkCls = ({ isActive }) =>
    `rounded-lg px-3 py-2 text-sm font-medium transition ${isActive ? 'bg-navy text-paper' : 'text-ink-soft hover:bg-paper-2 hover:text-ink'}`

  return (
    <div className="flex min-h-screen flex-col">
      <header className="sticky top-0 z-40 border-b border-line/80 bg-paper/85 backdrop-blur">
        <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
          <Logo />
          <nav className="hidden items-center gap-1 md:flex">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className={linkCls}>{l.label}</NavLink>
            ))}
          </nav>
          <div className="flex items-center gap-2">
            <StatusPill className="hidden sm:inline-flex" />
            <a href={REPO_URL} target="_blank" rel="noreferrer" className="hidden rounded-lg p-2 text-ink-soft hover:bg-paper-2 md:block" aria-label="GitHub repository">
              <Github size={18} />
            </a>
            <button className="rounded-lg p-2 md:hidden" onClick={() => setOpen((o) => !o)} aria-label="Menu">
              {open ? <X size={20} /> : <Menu size={20} />}
            </button>
          </div>
        </div>
        {open && (
          <nav className="flex flex-col gap-1 border-t border-line px-4 py-3 md:hidden">
            {LINKS.map((l) => (
              <NavLink key={l.to} to={l.to} end={l.end} className={linkCls}>{l.label}</NavLink>
            ))}
            <StatusPill className="mt-2 self-start" />
          </nav>
        )}
      </header>

      <main className="flex-1">
        <Outlet />
      </main>

      <footer className="border-t border-line bg-paper-2/60">
        <div className="mx-auto flex max-w-7xl flex-col gap-3 px-4 py-8 text-sm text-ink-soft sm:px-6 md:flex-row md:items-center md:justify-between">
          <div className="flex items-center gap-3">
            <Logo />
            <span className="text-muted">Sequential Penmanship Generation System</span>
          </div>
          <p className="max-w-xl text-xs leading-relaxed text-muted">
            For research and educational use only. Generated handwriting must not be used for impersonation,
            forgery or fraud.
          </p>
        </div>
      </footer>
    </div>
  )
}
