import { Link } from 'react-router-dom'
import {
  ArrowRight, ScanLine, Type, Waves, Gauge, PenTool, Languages, FlaskConical, BookOpen, ShieldAlert,
} from 'lucide-react'
import DenoiseCanvas from '../components/DenoiseCanvas'

const STATS = [
  { value: '1', unit: 'sample', label: 'Reference image needed to capture a writer’s style' },
  { value: '~35%', unit: '', label: 'Better style capture from the edge-based style encoder' },
  { value: '~90%', unit: '', label: 'OCR readability of generated text' },
  { value: '3+', unit: 'languages', label: 'Scripts evaluated: English, Chinese, Japanese' },
]

const PIPELINE = [
  {
    icon: ScanLine, title: 'Style encoder',
    text: 'A spatial encoder (ResNet-18) and a high-frequency encoder that reads the Laplacian edge map capture slant, spacing and stroke dynamics from one sample.',
  },
  {
    icon: Type, title: 'Text conditioning',
    text: 'Each character of the input text is turned into a 16×16 Unifont glyph, giving the model clean, language-agnostic content guidance.',
  },
  {
    icon: Waves, title: 'Diffusion generator',
    text: 'A UNet with style–content cross-attention denoises a latent step by step (DDIM / DDPM); the Stable Diffusion VAE decodes it to an image.',
  },
  {
    icon: Gauge, title: 'Evaluation',
    text: 'Generated samples are scored with OCR readability and visual similarity across 50+ samples per style.',
  },
]

const USES = [
  { icon: PenTool, title: 'Personalised handwriting', text: 'Render notes, cards or labels in a specific writing style.' },
  { icon: FlaskConical, title: 'Synthetic OCR data', text: 'Generate diverse labelled handwriting to train recognisers.' },
  { icon: Languages, title: 'Font & document stylisation', text: 'Explore writer-specific glyph shapes beyond fixed fonts.' },
  { icon: BookOpen, title: 'Low-shot research', text: 'Study style generalisation under extreme data scarcity.' },
]

export default function Home() {
  return (
    <>
      {/* hero */}
      <section className="relative overflow-hidden">
        <div className="dotgrid absolute inset-0 opacity-70" aria-hidden />
        <div className="relative mx-auto grid max-w-7xl items-center gap-12 px-4 py-16 sm:px-6 md:py-24 lg:grid-cols-[1.1fr_1fr]">
          <div className="rise">
            <span className="chip mb-6 border-accent/20 bg-accent-soft text-accent">Diffusion-based handwriting synthesis</span>
            <h1 className="font-display text-4xl leading-[1.05] font-semibold tracking-tight text-ink sm:text-5xl lg:text-6xl">
              One sample in.
              <br />
              <span className="font-hand text-6xl font-bold text-accent sm:text-7xl lg:text-8xl">your handwriting</span>
              <br />
              out.
            </h1>
            <p className="mt-6 max-w-xl text-lg leading-relaxed text-ink-soft">
              SYNGEN learns a writer’s slant, spacing and stroke style from a <strong className="text-ink">single reference image</strong>,
              then writes any text you type in that style using a conditional diffusion model.
            </p>
            <div className="mt-8 flex flex-wrap gap-3">
              <Link to="/studio" className="btn-primary px-5 py-3 text-base">
                Open the Studio <ArrowRight size={18} />
              </Link>
              <Link to="/architecture" className="btn-ghost px-5 py-3 text-base">How it works</Link>
            </div>
          </div>

          {/* pipeline preview card */}
          <div className="rise card relative p-5 sm:p-6" style={{ animationDelay: '120ms' }}>
            <div className="flex items-center justify-between">
              <span className="label">Generation pipeline</span>
              <span className="font-hand text-lg text-muted">one-shot</span>
            </div>
            <div className="mt-4 grid grid-cols-[auto_1fr] items-center gap-x-4 gap-y-5">
              <Step n="1" />
              <div>
                <p className="mb-1.5 text-xs font-semibold text-ink-soft">Style reference (one image)</p>
                <img src="/api/sample" onError={(e) => { e.currentTarget.style.display = 'none' }} alt="Handwritten sample reference" className="h-20 w-auto rounded-lg border border-line object-cover" />
              </div>
              <Step n="2" />
              <div>
                <p className="mb-1.5 text-xs font-semibold text-ink-soft">Content glyphs</p>
                <div className="flex flex-wrap gap-1">
                  {'syngen'.split('').map((c, i) => (
                    <span key={i} className="grid h-8 w-8 place-items-center rounded-md border border-ink/20 bg-white font-mono text-sm text-ink">{c}</span>
                  ))}
                </div>
              </div>
              <Step n="3" />
              <div>
                <p className="mb-1.5 text-xs font-semibold text-ink-soft">Reverse diffusion</p>
                <DenoiseCanvas word="syngen" />
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* stats */}
      <section className="border-y border-line bg-card/60">
        <div className="mx-auto grid max-w-7xl grid-cols-2 gap-px px-4 sm:px-6 lg:grid-cols-4">
          {STATS.map((s) => (
            <div key={s.label} className="px-2 py-8 sm:px-6">
              <div className="font-display text-4xl font-semibold text-navy">
                {s.value} <span className="font-sans text-sm font-medium text-muted">{s.unit}</span>
              </div>
              <p className="mt-2 text-sm leading-snug text-ink-soft">{s.label}</p>
            </div>
          ))}
        </div>
        <p className="mx-auto max-w-7xl px-4 pb-4 text-[11px] text-muted sm:px-6">Figures as reported in the project README.</p>
      </section>

      {/* how it works */}
      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6">
        <div className="max-w-2xl">
          <span className="label">System architecture</span>
          <h2 className="mt-2 font-display text-3xl font-semibold tracking-tight sm:text-4xl">Four stages from a photo to a page</h2>
          <p className="mt-3 text-ink-soft">
            The model jointly conditions on <em>what</em> to write (text) and <em>how</em> to write it (style), so the same reference
            can produce any word.
          </p>
        </div>
        <ol className="mt-10 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {PIPELINE.map((p, i) => (
            <li key={p.title} className="card relative p-6">
              <span className="absolute top-5 right-5 font-hand text-3xl text-line">{i + 1}</span>
              <span className="grid h-11 w-11 place-items-center rounded-xl bg-accent-soft text-accent"><p.icon size={22} /></span>
              <h3 className="mt-4 font-display text-lg font-semibold">{p.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-ink-soft">{p.text}</p>
            </li>
          ))}
        </ol>
        <div className="mt-8">
          <Link to="/architecture" className="inline-flex items-center gap-1.5 text-sm font-semibold text-accent hover:underline">
            See the full architecture <ArrowRight size={16} />
          </Link>
        </div>
      </section>

      {/* use cases */}
      <section className="ruled border-y border-line">
        <div className="mx-auto max-w-7xl px-4 py-20 pl-16 sm:px-6 sm:pl-20">
          <span className="label">Use cases</span>
          <h2 className="mt-2 font-display text-3xl font-semibold tracking-tight sm:text-4xl">Where one-shot handwriting helps</h2>
          <div className="mt-10 grid gap-x-10 gap-y-8 sm:grid-cols-2">
            {USES.map((u) => (
              <div key={u.title} className="flex gap-4">
                <u.icon className="mt-1 shrink-0 text-accent" size={22} />
                <div>
                  <h3 className="font-semibold">{u.title}</h3>
                  <p className="mt-1 text-sm text-ink-soft">{u.text}</p>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* CTA */}
      <section className="mx-auto max-w-7xl px-4 py-20 sm:px-6">
        <div className="card flex flex-col items-start justify-between gap-6 overflow-hidden bg-navy p-8 text-paper sm:p-12 md:flex-row md:items-center">
          <div>
            <h2 className="font-display text-3xl font-semibold">Try it with your own handwriting</h2>
            <p className="mt-2 max-w-xl text-paper/75">Upload a photo of one handwritten word, type a sentence, and watch the model write it.</p>
          </div>
          <Link to="/studio" className="btn bg-paper px-5 py-3 text-base text-navy hover:bg-white">
            Open the Studio <ArrowRight size={18} />
          </Link>
        </div>
        <p className="mt-6 flex items-start gap-2 text-xs text-muted">
          <ShieldAlert size={14} className="mt-0.5 shrink-0" />
          This project is intended strictly for research and educational purposes. Generated handwriting should not be used for
          impersonation, forgery, or fraudulent activities.
        </p>
      </section>
    </>
  )
}

function Step({ n }) {
  return <span className="grid h-7 w-7 place-items-center self-start rounded-full bg-navy text-xs font-bold text-paper">{n}</span>
}
