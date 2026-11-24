# SYNGEN Web Demo

A web interface for the SYNGEN handwriting synthesis model:

- **Overview**: what the project does, its key results, and use cases
- **Studio**: upload one handwriting sample, type text, and generate it in that style
- **Architecture**: model diagram, spec table (from `configs/IAM64.yml`), and commands
- **Gallery**: research figures from `assets/`, with a lightbox

```
frontend/   React 19 + Vite + Tailwind CSS (the UI)
server/     FastAPI wrapper around the model code in models/
run_demo.ps1  one-command launcher (Windows)
```

---

## Quick start (demo day)

You need Python 3.8+ and Node.js 18+.

```powershell
pip install -r server/requirements.txt          # API only (demo mode)
pip install -r server/requirements-model.txt    # + PyTorch/diffusers for the real model
.\run_demo.ps1                                   # builds the UI once, starts http://localhost:8000
```

Without the launcher, run these from the repo root:

```powershell
cd frontend; npm install; npm run build; cd ..
python server/app.py
```

Then open http://localhost:8000.

### Real model vs. demo mode

The API loads the trained UNet when these files are present (both are gitignored):

| File | Purpose |
|---|---|
| `model_zoo/One-DM-ckpt.pt` | trained UNet weights (state dict, or a dict with `model_state_dict`) |
| `data/unifont.pickle` | 16×16 content glyphs (a rendered fallback is used if it is missing) |

#### Where the weights came from (checked by SHA-256)

| File | Source | SHA-256 |
|---|---|---|
| `model_zoo/One-DM-ckpt.pt` (1.25 GB) | official One-DM release (HF copy `Dinh/One-DM`, identical to the authors' wisemodel.cn LFS object) | `f87a55369e00a7a642eccfc10d1f6dc31a842f555b9486a310d69c3d3e7c1888` |
| `model_zoo/stable-diffusion-v1-5/vae/` (335 MB) | `stable-diffusion-v1-5/stable-diffusion-v1-5` on HuggingFace | `a2b5134f4dbc140d9c11f11cba3233099e00af40f262f136c691fb7d38d2194c` |
| `data/unifont.pickle` (17 MB) | `blowing-up-groundhogs/vatrpp` (VATr authors), loaded with a numpy-only unpickler | `0804979068f0d169b343fbe0fe8d7ff478165d07a671fcf52e20f625db8e7f9f` |

These are the **official pretrained One-DM weights** (trained on IAM), not a checkpoint trained
by this project. If you train your own, point `SYNGEN_CKPT` at it.

On first run, torchvision downloads the ImageNet **ResNet-18** weights (45 MB) that the style encoder
in `models/fusion.py` starts from. If `model_zoo/stable-diffusion-v1-5/` is missing, the **VAE** (~330 MB)
is downloaded from HuggingFace as well.
Start the API once before demo day so both are already cached.
To work offline, put a copy at `model_zoo/stable-diffusion-v1-5/vae/`, or point
`SYNGEN_VAE` at a folder containing `vae/`.

If the checkpoint, PyTorch, or the VAE is missing, the API runs in **demo mode**. The
navbar shows `Demo mode`, the Studio shows a yellow banner, and every result is tagged
**"DEMO · font render, not model output"**. Style analysis (cleanup, 64 px style input,
Laplacian edge map, slant and stroke metrics) is real computation in both modes.

The navbar pill shows the current state: `Model ready · CUDA/CPU`, `Loading model…`,
`Demo mode`, or `API offline`.

### Speed

Each word is one diffusion run. On a laptop CPU, one DDIM step for a short word measured about
0.1 s, so 50 steps is roughly 5 s per word. A GPU is much faster. For a live CPU demo, use
20–30 DDIM steps and short sentences. The Studio shows live per-step progress.
DDPM always runs all 1000 steps.

### Settings (environment variables)

| Variable | Default |
|---|---|
| `SYNGEN_CKPT` | `model_zoo/One-DM-ckpt.pt` |
| `SYNGEN_VAE` | local `model_zoo/stable-diffusion-v1-5`, then `stable-diffusion-v1-5/stable-diffusion-v1-5` on HF |
| `SYNGEN_DEVICE` | `cuda` if available, else `cpu` |
| `SYNGEN_UNIFONT` | `data/unifont.pickle` |
| `SYNGEN_FORCE_DEMO` | set to `1` to skip model loading |
| `PORT` / `HOST` | `8000` / `127.0.0.1` |

---

## Development

Run the API and the Vite dev server (hot reload) side by side:

```powershell
python server/app.py              # terminal 1 -> :8000
cd frontend; npm run dev          # terminal 2 -> http://localhost:5173 (proxies /api to :8000)
```

### API

| Method | Path | Body | Returns |
|---|---|---|---|
| GET | `/api/health` | | mode, device, reason, charset |
| POST | `/api/analyze` | `file`, `enhance`, `fit` | cleaned / style / Laplace PNGs + metrics |
| POST | `/api/generate` | `file`, `text`, `method`, `steps`, `eta`, `seed`, `enhance`, `fit` | `job_id` |
| GET | `/api/jobs/{id}` | | status, progress, per-word PNGs, composed page |
| GET | `/api/sample` | | `test.jpg` |

### How a request is processed (`server/preprocess.py`, `server/engine.py`)

1. Find the ink in the photo: adaptive + Otsu threshold, then remove thin ruled and margin lines and dust.
2. Crop to the ink, whiten the background, resize to 64 px high (≤352 px wide).
3. Optionally apply the repo's enhancement chain (bilateral, CLAHE, unsharp, adaptive threshold).
4. Compute the Laplacian edge map for the high-frequency style encoder.
5. For each word: build Unifont glyph content, then run DDIM/DDPM with the repo's noise schedule and decode with the VAE.
6. Trim each word and lay the words out on a page.

---

## Troubleshooting

**`pip install` or the VAE download hangs with "handshake operation timed out"**
Some networks drop the new post-quantum TLS handshake used by Python 3.13+ (OpenSSL 3.5).
Create `openssl.cnf`:

```ini
openssl_conf = openssl_init
[openssl_init]
ssl_conf = ssl_sect
[ssl_sect]
system_default = system_default_sect
[system_default_sect]
Groups = X25519:secp256r1:secp384r1
```

Then set `$env:OPENSSL_CONF = "C:\path\to\openssl.cnf"` in the same terminal before running `pip` or `python server/app.py`.

**"Unsupported characters"**: the model only knows the IAM character set (letters,
digits and `!"#%&'()*+,-./:;?_`).

**Style reference tips**: use one word or a short line, photographed straight on, cropped
reasonably close. Dark ink on light paper works best. Ruled lines are removed automatically.

---

For research and educational use only. Generated handwriting must not be used for impersonation, forgery or fraud.
