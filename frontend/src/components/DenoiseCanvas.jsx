import { useEffect, useRef, useState } from 'react'

// Illustration of the reverse diffusion process: noise is gradually removed
// until the word appears. Purely decorative (drawn with a handwriting font).
const W = 300
const H = 72

export default function DenoiseCanvas({ word = 'handwriting', className = '' }) {
  const ref = useRef(null)
  const [t, setT] = useState(1000)

  useEffect(() => {
    const canvas = ref.current
    const ctx = canvas.getContext('2d', { willReadFrequently: true })
    let raf = 0
    let cancelled = false

    const run = async () => {
      try { await document.fonts.load('700 52px Caveat') } catch { /* ignore */ }
      if (cancelled) return
      // target image
      const off = document.createElement('canvas')
      off.width = W; off.height = H
      const o = off.getContext('2d')
      o.fillStyle = '#fff'; o.fillRect(0, 0, W, H)
      o.fillStyle = '#000'; o.font = '700 52px Caveat, cursive'
      o.textBaseline = 'middle'; o.textAlign = 'center'
      o.fillText(word, W / 2, H / 2 + 2)
      const target = o.getImageData(0, 0, W, H).data
      const clean = new Float32Array(W * H)
      for (let i = 0; i < W * H; i++) clean[i] = 1 - target[i * 4] / 255 // 1 = ink

      const noise = new Float32Array(W * H)
      const reseed = () => {
        for (let i = 0; i < W * H; i++) {
          // approx. gaussian
          noise[i] = (Math.random() + Math.random() + Math.random() - 1.5) * 1.15
        }
      }
      reseed()
      const img = ctx.createImageData(W, H)
      const reduced = window.matchMedia('(prefers-reduced-motion: reduce)').matches
      const DUR = 3200, HOLD = 1800
      let start = performance.now()

      const frame = (now) => {
        let el = now - start
        if (el > DUR + HOLD) { start = now; el = 0; reseed() }
        const p = reduced ? 1 : Math.min(1, el / DUR)
        // cosine-like schedule: alpha_bar goes 0 -> 1
        const ab = Math.pow(Math.sin((p * Math.PI) / 2), 3)
        const a = Math.sqrt(ab), b = Math.sqrt(1 - ab)
        const d = img.data
        for (let i = 0; i < W * H; i++) {
          const v = Math.max(0, Math.min(1, a * clean[i] + b * (noise[i] * 0.5 + 0.35)))
          // ink #1b2238 on paper #fffdf6
          d[i * 4] = 255 - v * (255 - 27)
          d[i * 4 + 1] = 253 - v * (253 - 34)
          d[i * 4 + 2] = 246 - v * (246 - 56)
          d[i * 4 + 3] = 255
        }
        ctx.putImageData(img, 0, 0)
        setT(Math.round((1 - p) * 999))
        if (!reduced) raf = requestAnimationFrame(frame)
      }
      raf = requestAnimationFrame(frame)
    }
    run()
    return () => { cancelled = true; cancelAnimationFrame(raf) }
  }, [word])

  return (
    <div className={className}>
      <canvas ref={ref} width={W} height={H} className="h-auto w-full rounded-lg border border-line" aria-label={`Illustration: noise denoising into the word ${word}`} />
      <div className="mt-2 flex items-center justify-between font-mono text-[11px] text-muted">
        <span>x<sub>t</sub> · t = {String(t).padStart(3, '0')}</span>
        <span>illustration</span>
      </div>
    </div>
  )
}
