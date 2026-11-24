import { useEffect, useRef, useState } from 'react'
import { ImagePlus, RotateCcw, Trash2 } from 'lucide-react'

const ACCEPT = ['image/png', 'image/jpeg', 'image/webp', 'image/bmp']

export default function Dropzone({ file, previewUrl, onFile, onClear, onSample, sampleLoading }) {
  const [drag, setDrag] = useState(false)
  const [err, setErr] = useState('')
  const input = useRef(null)

  const accept = (f) => {
    if (!f) return
    if (!ACCEPT.includes(f.type)) { setErr('Please choose a PNG, JPG, WEBP or BMP image.'); return }
    if (f.size > 15 * 1024 * 1024) { setErr('Image is larger than 15 MB.'); return }
    setErr('')
    onFile(f)
  }

  // paste an image from the clipboard anywhere on the page
  useEffect(() => {
    const onPaste = (e) => {
      const item = [...(e.clipboardData?.items || [])].find((i) => i.type.startsWith('image/'))
      if (item) accept(item.getAsFile())
    }
    window.addEventListener('paste', onPaste)
    return () => window.removeEventListener('paste', onPaste)
  })

  if (file) {
    return (
      <div className="rounded-xl border border-line bg-white p-3">
        <div className="grid h-36 place-items-center overflow-hidden rounded-lg bg-paper-2/60">
          <img src={previewUrl} alt="Style reference" className="max-h-36 w-auto object-contain" />
        </div>
        <div className="mt-3 flex items-center justify-between gap-2">
          <div className="min-w-0">
            <p className="truncate text-sm font-medium">{file.name}</p>
            <p className="text-xs text-muted">{(file.size / 1024).toFixed(0)} KB</p>
          </div>
          <div className="flex shrink-0 gap-1.5">
            <button className="btn-ghost px-2.5 py-1.5 text-xs" onClick={() => input.current?.click()}>
              <RotateCcw size={14} /> Replace
            </button>
            <button className="btn-ghost px-2.5 py-1.5 text-xs" onClick={onClear} aria-label="Remove image">
              <Trash2 size={14} />
            </button>
          </div>
        </div>
        <input ref={input} type="file" accept={ACCEPT.join(',')} hidden onChange={(e) => { accept(e.target.files?.[0]); e.target.value = '' }} />
      </div>
    )
  }

  return (
    <div>
      <button
        type="button"
        onClick={() => input.current?.click()}
        onDragOver={(e) => { e.preventDefault(); setDrag(true) }}
        onDragLeave={() => setDrag(false)}
        onDrop={(e) => { e.preventDefault(); setDrag(false); accept(e.dataTransfer.files?.[0]) }}
        className={`flex w-full flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-8 text-center transition ${
          drag ? 'border-accent bg-accent-soft' : 'border-line bg-white hover:border-accent/50 hover:bg-accent-soft/40'
        }`}
      >
        <span className="grid h-12 w-12 place-items-center rounded-full bg-accent-soft text-accent"><ImagePlus size={22} /></span>
        <span className="text-sm font-semibold">Drop a handwriting image here</span>
        <span className="text-xs text-muted">or click to browse · paste with Ctrl+V · PNG / JPG</span>
      </button>
      <input ref={input} type="file" accept={ACCEPT.join(',')} hidden onChange={(e) => { accept(e.target.files?.[0]); e.target.value = '' }} />
      <div className="mt-2 flex items-center justify-between">
        <span className="text-xs text-muted">Tip: one word or short line, cropped close.</span>
        <button type="button" onClick={onSample} disabled={sampleLoading} className="text-xs font-semibold text-accent hover:underline disabled:opacity-50">
          {sampleLoading ? 'Loading…' : 'Use sample image'}
        </button>
      </div>
      {err && <p className="mt-2 text-xs font-medium text-rose-700">{err}</p>}
    </div>
  )
}
