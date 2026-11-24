import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import {
  AlertTriangle, Check, Dice5, Download, History, Info, Loader2, ScanLine, Sparkles, Terminal, Type, Wand2,
} from 'lucide-react'
import Dropzone from '../components/Dropzone'
import { analyzeStyle, FALLBACK_CHARSET, getJob, getSampleFile, startGeneration } from '../api'
import { useHealth } from '../health'
import { statusOf } from '../components/StatusPill'
import { download, INKS, tint } from '../ink'

const PRESETS = ['Hello World', 'The quick brown fox jumps over the lazy dog', 'Minor Project 2026', 'Diffusion models write too!']
const randomSeed = () => Math.floor(Math.random() * 2 ** 31)

export default function Studio() {
  const { health, offline } = useHealth()
  const status = statusOf(health, offline)

  // reference image
  const [file, setFile] = useState(null)
  const [previewUrl, setPreviewUrl] = useState('')
  const [sampleLoading, setSampleLoading] = useState(false)
  const [enhance, setEnhance] = useState(true)
  const [fit, setFit] = useState('aspect')
  const [analysis, setAnalysis] = useState(null)
  const [analyzing, setAnalyzing] = useState(false)
  const [analyzeError, setAnalyzeError] = useState('')

  // text + sampling
  const [text, setText] = useState('Hello World')
  const [method, setMethod] = useState('ddim')
  const [steps, setSteps] = useState(50)
  const [eta, setEta] = useState(0)
  const [seed, setSeed] = useState(1001)
  const [lockSeed, setLockSeed] = useState(false)

  // generation
  const [job, setJob] = useState(null)
  const [genError, setGenError] = useState('')
  const [history, setHistory] = useState([])
  const [selected, setSelected] = useState(null)
  const pollRef = useRef(null)

  const charset = health?.charset || FALLBACK_CHARSET
  const badChars = useMemo(() => {
    const set = new Set(charset)
    return [...new Set([...text].filter((c) => !/\s/.test(c) && !set.has(c)))]
  }, [text, charset])
  const wordCount = text.split(/\s+/).filter(Boolean).length
  const maxWords = health?.max_words || 60
  const maxLen = health?.max_word_len || 24
  const tooLong = text.split(/\s+/).find((w) => w.length > maxLen)

  // ---------------------------------------------------------------- reference
  const setReference = (f) => {
    setFile(f)
    setAnalysis(null)
    setPreviewUrl((old) => { if (old) URL.revokeObjectURL(old); return f ? URL.createObjectURL(f) : '' })
  }

  const loadSample = async () => {
    setSampleLoading(true)
    try { setReference(await getSampleFile()) } catch (e) { setAnalyzeError(e.message) } finally { setSampleLoading(false) }
  }

  useEffect(() => {
    if (!file || offline) return
    let stale = false
    setAnalyzing(true)
    setAnalyzeError('')
    analyzeStyle(file, { enhance, fit })
      .then((r) => { if (!stale) setAnalysis(r) })
      .catch((e) => { if (!stale) { setAnalysis(null); setAnalyzeError(e.message) } })
      .finally(() => { if (!stale) setAnalyzing(false) })
    return () => { stale = true }
  }, [file, enhance, fit, offline])

  // --------------------------------------------------------------- generation
  const running = job && (job.status === 'queued' || job.status === 'running')
  const canGenerate = file && text.trim() && !badChars.length && !tooLong && wordCount <= maxWords && !running && !offline && !analyzeError

  const generate = useCallback(async () => {
    if (!canGenerate) return
    setGenError('')
    const useSeed = lockSeed ? seed : randomSeed()
    if (!lockSeed) setSeed(useSeed)
    try {
      const { job_id, seed: usedSeed } = await startGeneration(file, text.trim(), { enhance, fit, method, steps, eta, seed: useSeed })
      setJob({ id: job_id, status: 'queued', progress: 0, stage: 'Queued', words_done: 0, words_total: wordCount })
      setSelected(null)
      const poll = async () => {
        try {
          const j = await getJob(job_id)
          setJob(j)
          if (j.status === 'done') {
            setHistory((h) => [{ ...j, text: text.trim(), seed: usedSeed, at: new Date() }, ...h].slice(0, 12))
            setSelected(j.id)
            return
          }
          if (j.status === 'error') { setGenError(j.error || 'Generation failed.'); return }
          pollRef.current = setTimeout(poll, 400)
        } catch (e) {
          setGenError(e.message)
          setJob(null)
        }
      }
      poll()
    } catch (e) {
      setGenError(e.message)
    }
  }, [canGenerate, lockSeed, seed, file, text, enhance, fit, method, steps, eta, wordCount])

  useEffect(() => () => clearTimeout(pollRef.current), [])

  useEffect(() => {
    const onKey = (e) => { if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') generate() }
    window.addEventListener('keydown', onKey)
    return () => window.removeEventListener('keydown', onKey)
  }, [generate])

  const current = history.find((h) => h.id === selected)

  return (
    <div className="mx-auto max-w-7xl px-4 py-8 sm:px-6 sm:py-10">
      <div className="flex flex-col gap-3 md:flex-row md:items-end md:justify-between">
        <div>
          <span className="label">Studio</span>
          <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight sm:text-4xl">Generate handwriting</h1>
          <p className="mt-2 max-w-2xl text-ink-soft">Upload one handwriting sample, type your text, and let the diffusion model write it in that style.</p>
        </div>
      </div>

      <ModeBanner status={status} health={health} />

      <div className="mt-6 grid gap-6 lg:grid-cols-[400px_1fr]">
        {/* ------------------------------------------------ controls */}
        <div className="flex flex-col gap-5">
          <Panel n="1" title="Style reference" icon={ScanLine}>
            <Dropzone
              file={file} previewUrl={previewUrl} onFile={setReference} onClear={() => setReference(null)}
              onSample={loadSample} sampleLoading={sampleLoading}
            />
            <div className="mt-4 grid grid-cols-2 gap-3">
              <label className="flex cursor-pointer items-center justify-between gap-2 rounded-xl border border-line bg-white px-3 py-2">
                <span className="text-xs font-medium">Enhance</span>
                <Toggle checked={enhance} onChange={setEnhance} label="Enhance reference" />
              </label>
              <select value={fit} onChange={(e) => setFit(e.target.value)} className="rounded-xl border border-line bg-white px-3 py-2 text-xs font-medium" aria-label="Fit mode">
                <option value="aspect">Keep aspect (≤352px)</option>
                <option value="stretch">Stretch to 352×64</option>
              </select>
            </div>
            {analyzeError && <ErrorNote>{analyzeError}</ErrorNote>}
          </Panel>

          <Panel n="2" title="Text to write" icon={Type}>
            <textarea
              value={text} onChange={(e) => setText(e.target.value)} rows={4}
              placeholder="Type something…"
              className="w-full resize-y rounded-xl border border-line bg-white px-3.5 py-3 font-hand text-2xl leading-snug outline-none focus:border-accent"
            />
            <div className="mt-2 flex items-center justify-between text-xs text-muted">
              <span>{wordCount}/{maxWords} words · new line = new line on the page</span>
              {badChars.length === 0 && !tooLong && text.trim() && <span className="inline-flex items-center gap-1 text-emerald-700"><Check size={12} /> valid</span>}
            </div>
            {badChars.length > 0 && (
              <ErrorNote>Not in the model’s character set: {badChars.map((c) => <code key={c} className="mx-0.5 rounded bg-rose-100 px-1">{c}</code>)}</ErrorNote>
            )}
            {tooLong && <ErrorNote>“{tooLong.slice(0, 30)}” is longer than {maxLen} characters.</ErrorNote>}
            <div className="mt-3 flex flex-wrap gap-1.5">
              {PRESETS.map((p) => (
                <button key={p} onClick={() => setText(p)} className="chip hover:border-accent/40 hover:text-accent">{p}</button>
              ))}
            </div>
          </Panel>

          <Panel n="3" title="Sampling" icon={Wand2}>
            <div className="grid grid-cols-2 gap-1 rounded-xl bg-paper-2 p-1">
              {[['ddim', 'DDIM', 'fast'], ['ddpm', 'DDPM', '1000 steps']].map(([k, l, s]) => (
                <button key={k} onClick={() => setMethod(k)} className={`rounded-lg px-3 py-2 text-sm font-semibold transition ${method === k ? 'bg-white text-ink shadow-sm' : 'text-muted hover:text-ink'}`}>
                  {l} <span className="font-normal text-muted">· {s}</span>
                </button>
              ))}
            </div>
            {method === 'ddim' ? (
              <div className="mt-4 space-y-4">
                <Slider label="Steps" value={steps} min={10} max={100} step={5} onChange={setSteps} hint="more = slower, cleaner" />
                <Slider label="Eta (η)" value={eta} min={0} max={1} step={0.1} onChange={setEta} hint="0 = deterministic" format={(v) => v.toFixed(1)} />
              </div>
            ) : (
              <p className="mt-3 flex gap-2 text-xs text-ink-soft"><Info size={14} className="mt-0.5 shrink-0" /> DDPM runs all 1000 denoising steps per word. Use a GPU for this.</p>
            )}
            <div className="mt-4 flex items-center gap-2">
              <span className="w-12 text-xs font-medium text-ink-soft">Seed</span>
              <input
                type="number" value={seed} min={0} onChange={(e) => { setSeed(Math.max(0, parseInt(e.target.value || '0', 10))); setLockSeed(true) }}
                className="w-full min-w-0 rounded-lg border border-line bg-white px-3 py-1.5 font-mono text-sm"
              />
              <button className="btn-ghost px-2.5 py-1.5" onClick={() => setSeed(randomSeed())} aria-label="Random seed"><Dice5 size={16} /></button>
            </div>
            <label className="mt-2 flex items-center gap-2 text-xs text-ink-soft">
              <input type="checkbox" checked={lockSeed} onChange={(e) => setLockSeed(e.target.checked)} className="accent-accent" />
              Keep this seed (reproducible output)
            </label>
          </Panel>

          <button onClick={generate} disabled={!canGenerate} className="btn-primary w-full py-3.5 text-base">
            {running ? <><Loader2 size={18} className="animate-spin" /> Generating…</> : <><Sparkles size={18} /> Generate handwriting</>}
          </button>
          <p className="-mt-3 text-center text-[11px] text-muted">
            {!file ? 'Add a style reference to start' : offline ? 'API offline' : 'Ctrl + Enter to generate'}
          </p>
        </div>

        {/* ------------------------------------------------ results */}
        <div className="flex min-w-0 flex-col gap-5">
          <StyleAnalysis analysis={analysis} analyzing={analyzing} hasFile={!!file} />
          <OutputPanel job={job} result={current} error={genError} />
          {history.length > 0 && (
            <div className="card p-5">
              <h3 className="flex items-center gap-2 text-sm font-semibold"><History size={16} /> This session</h3>
              <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                {history.map((h) => (
                  <button key={h.id} onClick={() => { setSelected(h.id); setJob(null) }}
                    className={`rounded-xl border p-2 text-left transition ${selected === h.id ? 'border-accent bg-accent-soft/50' : 'border-line bg-white hover:border-ink/20'}`}>
                    <div className="grid h-14 place-items-center overflow-hidden rounded-md bg-white"><img src={h.page} alt="" className="max-h-14 w-auto" /></div>
                    <p className="mt-1.5 truncate text-xs font-medium">{h.text}</p>
                    <p className="text-[11px] text-muted">seed {h.seed} · {h.params.method.toUpperCase()} · {h.elapsed}s {h.mode === 'demo' && '· demo'}</p>
                  </button>
                ))}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

/* ------------------------------------------------------------------ pieces */

function ModeBanner({ status, health }) {
  if (status.key === 'model') return null
  if (status.key === 'offline') {
    return (
      <div className="mt-6 flex gap-3 rounded-2xl border border-rose-200 bg-rose-50 p-4 text-sm text-rose-900">
        <Terminal size={18} className="mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold">The SYNGEN API is not running.</p>
          <p className="mt-1">Start it from the repository root, then this page connects automatically:</p>
          <code className="mt-2 inline-block rounded-lg bg-white/70 px-2 py-1 font-mono text-xs">python server/app.py</code>
        </div>
      </div>
    )
  }
  if (status.key === 'demo') {
    return (
      <div className="mt-6 flex gap-3 rounded-2xl border border-amber-200 bg-amber-50 p-4 text-sm text-amber-950">
        <AlertTriangle size={18} className="mt-0.5 shrink-0" />
        <div>
          <p className="font-semibold">Demo mode: outputs are typeset with a font, not generated by the model.</p>
          <p className="mt-1 text-amber-900/80">{health?.reason} Style analysis below is real. Place the trained weights at <code className="font-mono">model_zoo/One-DM-ckpt.pt</code> and restart the API to generate with the model.</p>
        </div>
      </div>
    )
  }
  if (status.key === 'loading') {
    return (
      <div className="mt-6 flex items-center gap-3 rounded-2xl border border-sky-200 bg-sky-50 p-4 text-sm text-sky-950">
        <Loader2 size={18} className="animate-spin" /> {health?.reason || 'Loading model…'}
      </div>
    )
  }
  return null
}

function Panel({ n, title, icon: Icon, children }) {
  return (
    <section className="card p-5">
      <h2 className="mb-4 flex items-center gap-2.5 text-sm font-semibold">
        <span className="grid h-6 w-6 place-items-center rounded-full bg-navy text-[11px] font-bold text-paper">{n}</span>
        {title}
        <Icon size={15} className="ml-auto text-muted" />
      </h2>
      {children}
    </section>
  )
}

function Toggle({ checked, onChange, label }) {
  return (
    <button type="button" role="switch" aria-checked={checked} aria-label={label} onClick={() => onChange(!checked)}
      className={`relative h-5 w-9 rounded-full transition ${checked ? 'bg-accent' : 'bg-line'}`}>
      <span className={`absolute top-0.5 h-4 w-4 rounded-full bg-white shadow transition-all ${checked ? 'left-[18px]' : 'left-0.5'}`} />
    </button>
  )
}

function Slider({ label, value, min, max, step, onChange, hint, format = (v) => v }) {
  return (
    <div>
      <div className="mb-1 flex items-center justify-between text-xs">
        <span className="font-medium text-ink-soft">{label} <span className="font-normal text-muted">· {hint}</span></span>
        <span className="font-mono font-semibold">{format(value)}</span>
      </div>
      <input type="range" className="w-full" value={value} min={min} max={max} step={step} onChange={(e) => onChange(Number(e.target.value))} aria-label={label} />
    </div>
  )
}

function ErrorNote({ children }) {
  return <p className="mt-2 flex gap-1.5 rounded-lg bg-rose-50 px-2.5 py-2 text-xs font-medium text-rose-800"><AlertTriangle size={14} className="mt-px shrink-0" /><span>{children}</span></p>
}

function StyleAnalysis({ analysis, analyzing, hasFile }) {
  if (!hasFile) {
    return (
      <div className="card dotgrid grid min-h-40 place-items-center p-8 text-center">
        <div>
          <p className="font-hand text-3xl text-ink-soft">Style analysis appears here</p>
          <p className="mt-1 text-sm text-muted">Add a reference image to see what the style encoder receives.</p>
        </div>
      </div>
    )
  }
  const m = analysis?.metrics
  return (
    <section className="card relative overflow-hidden p-5">
      {analyzing && <div className="shimmer pointer-events-none absolute inset-0" />}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold">Style analysis</h3>
        {analyzing && <span className="flex items-center gap-1.5 text-xs text-muted"><Loader2 size={13} className="animate-spin" /> processing</span>}
      </div>
      {analysis ? (
        <>
          <div className="mt-4 grid gap-4 md:grid-cols-2">
            <Figure label="Cleaned & cropped" sub="ruled lines and background removed">
              <img src={analysis.cleaned} alt="Cleaned reference" className="max-h-28 w-auto" />
            </Figure>
            <div className="grid gap-4">
              <Figure label="Style input · 64 px high" sub="spatial style encoder">
                <img src={analysis.style} alt="Model style input" className="h-16 w-auto max-w-full" />
              </Figure>
              <Figure label="Laplacian edge map" sub="high-frequency style encoder" dark>
                <img src={analysis.laplace} alt="Laplace edge map" className="h-16 w-auto max-w-full" />
              </Figure>
            </div>
          </div>
          <div className="mt-4 grid grid-cols-2 gap-2 sm:grid-cols-4">
            <Metric label="Slant" value={`${m.slant_deg > 0 ? '+' : ''}${m.slant_deg}°`} hint={Math.abs(m.slant_deg) < 4 ? 'upright' : m.slant_deg > 0 ? 'right-leaning' : 'left-leaning'} />
            <Metric label="Stroke width" value={`${m.stroke_px}px`} hint="at 64 px height" />
            <Metric label="Ink density" value={`${m.ink_density}%`} hint="of style image" />
            <Metric label="Style width" value={`${m.style_width_px}px`} hint={m.truncated ? 'cropped to 352' : 'max 352'} />
          </div>
          {m.truncated && <p className="mt-3 text-xs text-amber-800">The reference is wider than 352 px at 64 px height, so only the first part is used. A single word works best.</p>}
        </>
      ) : !analyzing ? <p className="mt-3 text-sm text-muted">No analysis yet.</p> : <div className="mt-4 h-40" />}
    </section>
  )
}

function Figure({ label, sub, children, dark }) {
  return (
    <figure>
      <div className={`grid min-h-20 place-items-center overflow-hidden rounded-lg border border-line p-2 ${dark ? 'bg-ink' : 'bg-white'}`}>{children}</div>
      <figcaption className="mt-1.5 text-xs"><span className="font-semibold">{label}</span> <span className="text-muted">· {sub}</span></figcaption>
    </figure>
  )
}

function Metric({ label, value, hint }) {
  return (
    <div className="rounded-xl border border-line bg-white px-3 py-2.5">
      <p className="text-[11px] font-medium text-muted">{label}</p>
      <p className="font-display text-xl font-semibold">{value}</p>
      <p className="text-[11px] text-muted">{hint}</p>
    </div>
  )
}

function OutputPanel({ job, result, error }) {
  const [ink, setInk] = useState(INKS[1].hex)
  const [paper, setPaper] = useState('ruled')
  const [tinted, setTinted] = useState('')
  const [tintedWords, setTintedWords] = useState([])
  const page = result?.page

  useEffect(() => {
    if (!result) return
    let stale = false
    tint(result.page, ink, null, 2).then((u) => !stale && setTinted(u))
    Promise.all(result.words.map((w) => tint(w.image, ink, null, 2))).then((u) => !stale && setTintedWords(u))
    return () => { stale = true }
  }, [result, ink])

  const running = job && (job.status === 'queued' || job.status === 'running')

  if (error) {
    return <section className="card p-5"><h3 className="text-sm font-semibold">Output</h3><ErrorNote>{error}</ErrorNote></section>
  }
  if (running) {
    const pct = Math.round((job.progress || 0) * 100)
    return (
      <section className="card p-6">
        <div className="flex items-center justify-between text-sm">
          <span className="flex items-center gap-2 font-semibold"><Loader2 size={16} className="animate-spin text-accent" /> {job.stage}</span>
          <span className="font-mono text-muted">{pct}%</span>
        </div>
        <div className="mt-3 h-2 overflow-hidden rounded-full bg-paper-2">
          <div className="h-full rounded-full bg-accent transition-[width] duration-300" style={{ width: `${Math.max(3, pct)}%` }} />
        </div>
        <p className="mt-2 text-xs text-muted">{job.words_done}/{job.words_total} words · {job.elapsed ?? 0}s elapsed</p>
      </section>
    )
  }
  if (!result) {
    return (
      <section className="card ruled grid min-h-56 place-items-center p-8 pl-16 text-center">
        <div>
          <p className="font-hand text-4xl text-ink-soft/70">Your handwriting will appear here</p>
          <p className="mt-2 text-sm text-muted">Generated words are stitched into a page you can recolour and download.</p>
        </div>
      </section>
    )
  }

  const isDemo = result.mode === 'demo'
  const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-')
  return (
    <section className="card rise overflow-hidden">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-line px-5 py-3">
        <div className="flex items-center gap-2">
          <h3 className="text-sm font-semibold">Output</h3>
          {isDemo
            ? <span className="rounded-full bg-amber-100 px-2 py-0.5 text-[11px] font-bold text-amber-900">DEMO · font render, not model output</span>
            : <span className="rounded-full bg-emerald-100 px-2 py-0.5 text-[11px] font-bold text-emerald-800">Generated by SYNGEN</span>}
        </div>
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-1.5" role="radiogroup" aria-label="Ink colour">
            {INKS.map((i) => (
              <button key={i.hex} title={i.name} aria-label={i.name} onClick={() => setInk(i.hex)}
                className={`h-5 w-5 rounded-full ring-offset-2 transition ${ink === i.hex ? 'ring-2 ring-accent' : ''}`} style={{ background: i.hex }} />
            ))}
          </div>
          <select value={paper} onChange={(e) => setPaper(e.target.value)} className="rounded-lg border border-line bg-white px-2 py-1 text-xs" aria-label="Paper">
            <option value="ruled">Ruled paper</option>
            <option value="plain">Plain</option>
            <option value="raw">Raw output</option>
          </select>
        </div>
      </div>

      <div className={`overflow-x-auto p-6 ${paper === 'ruled' ? 'ruled pl-16' : 'bg-white'}`}>
        {paper === 'raw'
          ? <img src={page} alt="Raw generated output" className="max-w-none" style={{ height: result.page_size[1] * 1.25 }} />
          : tinted && <img src={tinted} alt="Generated handwriting" className="max-w-none" style={{ height: result.page_size[1] * 1.25 }} />}
      </div>

      <div className="border-t border-line px-5 py-4">
        <p className="label mb-3">Individual words</p>
        <div className="flex flex-wrap gap-2">
          {result.words.map((w, i) => (
            <figure key={i} className="rounded-lg border border-line bg-white p-1.5">
              <img src={tintedWords[i] || w.image} alt={w.word} className="h-10 w-auto" />
              <figcaption className="mt-0.5 text-center font-mono text-[10px] text-muted">{w.word}</figcaption>
            </figure>
          ))}
        </div>
      </div>

      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line bg-paper-2/40 px-5 py-3">
        <p className="text-xs text-muted">
          seed <span className="font-mono">{result.params.seed}</span> · {result.params.method.toUpperCase()}
          {result.params.method === 'ddim' && ` · ${result.params.steps} steps · η ${result.params.eta}`} · {result.elapsed}s
        </p>
        <div className="flex gap-2">
          <button className="btn-ghost px-3 py-1.5 text-xs" onClick={async () => download(await tint(page, ink, '#ffffff', 2), `syngen-${stamp}.png`)}>
            <Download size={14} /> PNG
          </button>
          <button className="btn-ghost px-3 py-1.5 text-xs" onClick={() => tinted && download(tinted, `syngen-${stamp}-transparent.png`)}>
            <Download size={14} /> Transparent PNG
          </button>
        </div>
      </div>
    </section>
  )
}
