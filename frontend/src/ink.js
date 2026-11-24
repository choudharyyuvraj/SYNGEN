// Re-colour a grayscale handwriting image: dark pixels become `ink`,
// light pixels become transparent (or white when `background` is set).
const cache = new Map()

function loadImage(src) {
  return new Promise((resolve, reject) => {
    const img = new Image()
    img.onload = () => resolve(img)
    img.onerror = reject
    img.src = src
  })
}

function hexToRgb(hex) {
  const n = parseInt(hex.replace('#', ''), 16)
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255]
}

export async function tint(src, ink = '#111111', background = null, scale = 1) {
  const key = `${src.length}:${src.slice(-64)}|${ink}|${background}|${scale}`
  if (cache.has(key)) return cache.get(key)
  const img = await loadImage(src)
  const w = Math.round(img.naturalWidth * scale)
  const h = Math.round(img.naturalHeight * scale)
  const c = document.createElement('canvas')
  c.width = w; c.height = h
  const ctx = c.getContext('2d')
  ctx.imageSmoothingQuality = 'high'
  ctx.drawImage(img, 0, 0, w, h)
  const data = ctx.getImageData(0, 0, w, h)
  const d = data.data
  const [r, g, b] = hexToRgb(ink)
  const bg = background ? hexToRgb(background) : null
  for (let i = 0; i < d.length; i += 4) {
    const gray = (d[i] + d[i + 1] + d[i + 2]) / 3
    // soft threshold keeps anti-aliasing but drops paper noise
    let a = Math.min(1, Math.max(0, (235 - gray) / 175))
    if (bg) {
      d[i] = bg[0] + (r - bg[0]) * a
      d[i + 1] = bg[1] + (g - bg[1]) * a
      d[i + 2] = bg[2] + (b - bg[2]) * a
      d[i + 3] = 255
    } else {
      d[i] = r; d[i + 1] = g; d[i + 2] = b; d[i + 3] = Math.round(a * 255)
    }
  }
  ctx.putImageData(data, 0, 0)
  const url = c.toDataURL('image/png')
  if (cache.size > 60) cache.clear()
  cache.set(key, url)
  return url
}

export function download(url, filename) {
  const a = document.createElement('a')
  a.href = url
  a.download = filename
  document.body.appendChild(a)
  a.click()
  a.remove()
}

export const INKS = [
  { name: 'Graphite', hex: '#1f1f24' },
  { name: 'Royal blue', hex: '#2336a8' },
  { name: 'Navy', hex: '#1e2a5a' },
  { name: 'Sepia', hex: '#5b3a1e' },
]
