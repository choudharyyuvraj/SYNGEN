// Small client for the FastAPI server in /server.
const BASE = import.meta.env.VITE_API_URL || ''

async function request(path, options = {}) {
  let res
  try {
    res = await fetch(BASE + path, options)
  } catch {
    throw new Error('Cannot reach the SYNGEN API. Start it with: python server/app.py')
  }
  let body = null
  try { body = await res.json() } catch { /* non-JSON */ }
  if (!res.ok) {
    const detail = body?.detail
    const msg = Array.isArray(detail) ? detail.map((d) => d.msg).join('; ') : detail
    throw new Error(msg || `Request failed (${res.status})`)
  }
  return body
}

function styleForm(file, { enhance = true, fit = 'aspect' } = {}) {
  const fd = new FormData()
  fd.append('file', file)
  fd.append('enhance', String(enhance))
  fd.append('fit', fit)
  return fd
}

export const getHealth = () => request('/api/health')

export const analyzeStyle = (file, opts) =>
  request('/api/analyze', { method: 'POST', body: styleForm(file, opts) })

export function startGeneration(file, text, opts) {
  const fd = styleForm(file, opts)
  fd.append('text', text)
  fd.append('method', opts.method)
  fd.append('steps', String(opts.steps))
  fd.append('eta', String(opts.eta))
  fd.append('seed', String(opts.seed))
  return request('/api/generate', { method: 'POST', body: fd })
}

export const getJob = (id) => request(`/api/jobs/${id}`)

export async function getSampleFile() {
  const res = await fetch(BASE + '/api/sample')
  if (!res.ok) throw new Error('Could not load the sample image.')
  const blob = await res.blob()
  return new File([blob], 'test.jpg', { type: blob.type || 'image/jpeg' })
}

// Default charset (mirrors data_loader/loader.py) used before /api/health answers.
export const FALLBACK_CHARSET =
  "!\"#%&'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZ_abcdefghijklmnopqrstuvwxyz"
