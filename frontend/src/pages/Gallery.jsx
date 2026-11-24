import { useEffect, useState } from 'react'
import { ChevronLeft, ChevronRight, Maximize2, X } from 'lucide-react'
import { CREDIT, FIGURES } from '../figures'

const CATS = ['All', 'Method', 'English', 'Multilingual', 'Comparisons']

export default function Gallery() {
  const [cat, setCat] = useState('All')
  const [open, setOpen] = useState(-1)
  const list = cat === 'All' ? FIGURES : FIGURES.filter((f) => f.cat === cat)

  useEffect(() => {
    if (open < 0) return
    const onKey = (e) => {
      if (e.key === 'Escape') setOpen(-1)
      if (e.key === 'ArrowRight') setOpen((i) => (i + 1) % list.length)
      if (e.key === 'ArrowLeft') setOpen((i) => (i - 1 + list.length) % list.length)
    }
    window.addEventListener('keydown', onKey)
    document.body.style.overflow = 'hidden'
    return () => { window.removeEventListener('keydown', onKey); document.body.style.overflow = '' }
  }, [open, list.length])

  const cur = list[open]

  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <span className="label">Gallery</span>
      <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight sm:text-4xl">Results across styles and scripts</h1>
      <p className="mt-2 max-w-2xl text-ink-soft">{CREDIT}</p>

      <div className="mt-6 flex flex-wrap gap-2" role="tablist">
        {CATS.map((c) => (
          <button key={c} role="tab" aria-selected={cat === c} onClick={() => setCat(c)}
            className={`rounded-full px-4 py-1.5 text-sm font-medium transition ${cat === c ? 'bg-navy text-paper' : 'border border-line bg-card text-ink-soft hover:text-ink'}`}>
            {c}
          </button>
        ))}
      </div>

      <div className="mt-8 columns-1 gap-5 md:columns-2 xl:columns-3">
        {list.map((f, i) => (
          <figure key={f.file} className="card group mb-5 break-inside-avoid overflow-hidden">
            <button onClick={() => setOpen(i)} className="relative block w-full bg-white p-3" aria-label={`Open ${f.title}`}>
              <img src={f.src} alt={f.title} loading="lazy" className="w-full" />
              <span className="absolute top-3 right-3 rounded-lg bg-ink/70 p-1.5 text-paper opacity-0 transition group-hover:opacity-100"><Maximize2 size={14} /></span>
            </button>
            <figcaption className="border-t border-line px-4 py-3">
              <div className="flex items-center justify-between gap-2">
                <p className="font-semibold">{f.title}</p>
                <span className="chip py-0.5 text-[10px]">{f.cat}</span>
              </div>
              <p className="mt-1 text-sm text-ink-soft">{f.text}</p>
            </figcaption>
          </figure>
        ))}
      </div>

      {cur && (
        <div className="fixed inset-0 z-50 flex flex-col bg-ink/90 p-4 backdrop-blur-sm sm:p-8" onClick={() => setOpen(-1)} role="dialog" aria-modal="true" aria-label={cur.title}>
          <div className="flex items-center justify-between text-paper">
            <div>
              <p className="font-display text-lg font-semibold">{cur.title}</p>
              <p className="text-sm text-paper/70">{cur.text}</p>
            </div>
            <button className="rounded-lg p-2 hover:bg-white/10" aria-label="Close"><X /></button>
          </div>
          <div className="relative mt-4 flex min-h-0 flex-1 items-center justify-center">
            <img src={cur.src} alt={cur.title} className="max-h-full max-w-full rounded-lg bg-white p-2" onClick={(e) => e.stopPropagation()} />
            <button className="absolute left-0 rounded-full bg-white/10 p-2 text-paper hover:bg-white/20" aria-label="Previous"
              onClick={(e) => { e.stopPropagation(); setOpen((i) => (i - 1 + list.length) % list.length) }}><ChevronLeft /></button>
            <button className="absolute right-0 rounded-full bg-white/10 p-2 text-paper hover:bg-white/20" aria-label="Next"
              onClick={(e) => { e.stopPropagation(); setOpen((i) => (i + 1) % list.length) }}><ChevronRight /></button>
          </div>
        </div>
      )}
    </div>
  )
}
