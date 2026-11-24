import { useState } from 'react'
import { ArrowDown, ArrowRight, Check, Copy } from 'lucide-react'
import { CREDIT, figure } from '../figures'

const SPECS = [
  ['Denoiser', 'UNet, 512-d embeddings, 1 res-block, 4 attention heads'],
  ['Latent space', '4 channels at 1/8 resolution (Stable Diffusion v1.5 VAE)'],
  ['Style input', 'Grayscale 64 px high, up to 352 px wide, plus Laplacian edge map'],
  ['Content input', 'One 16×16 Unifont glyph per character'],
  ['Noise schedule', '1000 steps, linear β from 1e-4 to 0.02'],
  ['Samplers', 'DDIM (50 steps default, η = 0) or full DDPM'],
  ['Training', 'AdamW, lr 1e-4, batch 96, gradient clip 5.0, seed 1001'],
  ['Dataset', 'IAM handwriting (64 px), train and test writer splits'],
]

const COMMANDS = [
  ['Run the web demo', 'python server/app.py'],
  ['Quick subset training (1%)', 'python custom_5percent/train_5percent.py --cfg configs/IAM64_5percent.yml --device cpu --subset_ratio 0.01'],
  ['Full training', 'python train.py --cfg configs/IAM64_scratch.yml --device cuda'],
  ['Test on IAM', 'python custom_5percent/test_iam.py --cfg configs/IAM64_5percent.yml --device cpu'],
  ['Single image test', 'python custom_5percent/test_single.py --style_image test.jpg --one_dm model_zoo/One-DM-ckpt.pt --device cpu'],
]

const STRUCTURE = [
  ['models/', 'UNet, diffusion, style-content fusion, ResNet encoders, losses'],
  ['data_loader/', 'IAM dataset, style/Laplace reference loading, Unifont content glyphs'],
  ['trainer/', 'Distributed and single-device training loops'],
  ['configs/', 'YAML configs: full, scratch, fine-tune, 5% subset'],
  ['custom_5percent/', 'Subset training and quick testing scripts'],
  ['server/', 'FastAPI wrapper used by this web app'],
  ['frontend/', 'This React interface'],
]

export default function Architecture() {
  return (
    <div className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
      <span className="label">Architecture</span>
      <h1 className="mt-1 font-display text-3xl font-semibold tracking-tight sm:text-4xl">How SYNGEN writes</h1>
      <p className="mt-2 max-w-3xl text-ink-soft">
        The generator is a latent diffusion model conditioned on two things: the <strong>content</strong> (which characters to
        write) and the <strong>style</strong> (how one particular writer writes). Style comes from a single reference image,
        read twice: once as pixels and once as a high-frequency edge map that isolates stroke detail.
      </p>

      {/* diagram */}
      <section className="card dotgrid mt-8 overflow-x-auto p-6 sm:p-8">
        <div className="grid min-w-[760px] grid-cols-[1fr_auto_1.1fr_auto_1.3fr] items-center gap-4">
          {/* inputs */}
          <div className="flex flex-col gap-4">
            <Node tone="amber" title="Style reference" sub="one image of handwriting" />
            <div className="grid grid-cols-2 gap-2">
              <Node small tone="amber" title="Spatial" sub="64×W grayscale" />
              <Node small tone="amber" title="High-freq" sub="Laplacian edges" />
            </div>
            <Node tone="green" title="Text content" sub="Unifont 16×16 glyphs" />
          </div>
          <Arrow />
          {/* encoders + fusion */}
          <div className="flex flex-col gap-4">
            <Node tone="blue" title="Style encoders" sub="ResNet-18 (dilated) + transformer · gated" />
            <Node tone="blue" title="Content encoder" sub="glyph sequence → embeddings" />
            <Node tone="violet" title="Style–content fusion" sub="cross-attention (Q = content, K/V = style) → self-attention" />
          </div>
          <Arrow />
          {/* diffusion */}
          <div className="flex flex-col items-stretch gap-3">
            <Node tone="ink" title="UNet denoiser" sub="predicts noise ε at each step t, conditioned on fused guidance" />
            <div className="flex items-center justify-center gap-2 font-mono text-xs text-ink-soft">
              x<sub>T</sub> ~ 𝒩(0, I) <ArrowRight size={14} /> … <ArrowRight size={14} /> x<sub>0</sub>
              <span className="rounded bg-paper-2 px-1.5 py-0.5">DDIM / DDPM</span>
            </div>
            <ArrowDown className="mx-auto text-muted" size={18} />
            <Node tone="ink" title="VAE decoder" sub="SD v1.5 · latent 4×8×4L → 64×32L image" />
            <ArrowDown className="mx-auto text-muted" size={18} />
            <div className="rounded-xl border border-line bg-white px-4 py-3 text-center font-hand text-3xl text-navy">handwritten word</div>
          </div>
        </div>
      </section>

      <div className="mt-8 grid gap-6 lg:grid-cols-2">
        <section className="card p-6">
          <h2 className="font-display text-xl font-semibold">Model specification</h2>
          <dl className="mt-4 divide-y divide-line">
            {SPECS.map(([k, v]) => (
              <div key={k} className="grid grid-cols-[130px_1fr] gap-4 py-2.5 text-sm">
                <dt className="font-medium text-ink-soft">{k}</dt>
                <dd>{v}</dd>
              </div>
            ))}
          </dl>
          <p className="mt-3 text-xs text-muted">From <code>configs/IAM64.yml</code>, <code>models/diffusion.py</code> and <code>data_loader/loader.py</code>.</p>
        </section>

        <section className="card p-6">
          <h2 className="font-display text-xl font-semibold">What the web demo does per request</h2>
          <ol className="mt-4 space-y-3 text-sm">
            {[
              'Find the ink in the uploaded photo, remove ruled lines and background, and crop tightly.',
              'Resize to 64 px height (≤ 352 px wide) and apply the repo’s enhancement chain (bilateral filter, CLAHE, unsharp mask, adaptive threshold).',
              'Compute the Laplacian edge map for the high-frequency style encoder.',
              'For every word: build the glyph sequence, sample a latent with DDIM/DDPM, then decode with the VAE.',
              'Trim each word, lay out words and lines on a page, and return PNGs to the browser.',
            ].map((s, i) => (
              <li key={i} className="flex gap-3">
                <span className="grid h-6 w-6 shrink-0 place-items-center rounded-full bg-accent-soft text-xs font-bold text-accent">{i + 1}</span>
                <span className="text-ink-soft">{s}</span>
              </li>
            ))}
          </ol>
        </section>
      </div>

      {figure('overview_v2.png') && (
        <figure className="card mt-8 overflow-hidden">
          <img src={figure('overview_v2.png')} alt="One-DM model overview" className="w-full bg-white p-4" />
          <figcaption className="border-t border-line px-5 py-3 text-sm text-ink-soft">Model overview. {CREDIT}</figcaption>
        </figure>
      )}

      <div className="mt-8 grid gap-6 lg:grid-cols-[1.4fr_1fr]">
        <section className="card p-6">
          <h2 className="font-display text-xl font-semibold">Commands</h2>
          <div className="mt-4 space-y-3">
            {COMMANDS.map(([label, cmd]) => <Command key={label} label={label} cmd={cmd} />)}
          </div>
        </section>
        <section className="card p-6">
          <h2 className="font-display text-xl font-semibold">Repository layout</h2>
          <ul className="mt-4 space-y-2.5 text-sm">
            {STRUCTURE.map(([k, v]) => (
              <li key={k} className="grid grid-cols-[130px_1fr] gap-3">
                <code className="font-mono text-xs font-semibold text-navy">{k}</code>
                <span className="text-ink-soft">{v}</span>
              </li>
            ))}
          </ul>
          <h3 className="mt-6 text-sm font-semibold">Tech stack</h3>
          <div className="mt-2 flex flex-wrap gap-1.5">
            {['PyTorch', 'Diffusers (SD VAE)', 'OpenCV', 'FastAPI', 'React 19', 'Vite', 'Tailwind CSS'].map((t) => <span key={t} className="chip">{t}</span>)}
          </div>
        </section>
      </div>
    </div>
  )
}

const TONES = {
  amber: 'border-amber-200 bg-amber-50',
  green: 'border-emerald-200 bg-emerald-50',
  blue: 'border-indigo-200 bg-indigo-50',
  violet: 'border-violet-200 bg-violet-50',
  ink: 'border-navy/20 bg-white',
}

function Node({ title, sub, tone, small }) {
  return (
    <div className={`rounded-xl border px-3 ${small ? 'py-2' : 'py-3'} ${TONES[tone]}`}>
      <p className={`${small ? 'text-xs' : 'text-sm'} font-semibold`}>{title}</p>
      <p className={`${small ? 'text-[10px]' : 'text-xs'} text-ink-soft`}>{sub}</p>
    </div>
  )
}

function Arrow() {
  return <ArrowRight className="text-muted" size={22} />
}

function Command({ label, cmd }) {
  const [copied, setCopied] = useState(false)
  return (
    <div>
      <p className="mb-1 text-xs font-medium text-ink-soft">{label}</p>
      <div className="flex items-start gap-2 rounded-lg bg-ink px-3 py-2">
        <code className="min-w-0 flex-1 font-mono text-xs leading-relaxed break-all text-paper">{cmd}</code>
        <button
          onClick={() => { navigator.clipboard?.writeText(cmd); setCopied(true); setTimeout(() => setCopied(false), 1200) }}
          className="shrink-0 rounded p-1 text-paper/70 hover:bg-white/10 hover:text-paper" aria-label="Copy command"
        >
          {copied ? <Check size={14} /> : <Copy size={14} />}
        </button>
      </div>
    </div>
  )
}
