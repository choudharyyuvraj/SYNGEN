"""Model engine for the SYNGEN web demo.

Loads the repo's UNet + Diffusion + Stable-Diffusion VAE once, in the
background, and runs generation jobs one at a time on a worker thread so the
frontend can poll progress.

If the checkpoint, PyTorch or the VAE is unavailable, the engine falls back to
DEMO mode: words are typeset with a handwriting-like system font so the whole
UI flow still works. Every demo result is flagged ``mode = "demo"`` and the UI
labels it clearly - it is NOT model output.

Environment variables (all optional):
  SYNGEN_CKPT       path to the trained UNet weights   (default model_zoo/One-DM-ckpt.pt)
  SYNGEN_CFG        model config                        (default configs/IAM64.yml)
  SYNGEN_VAE        HF repo id or local folder with a ``vae`` subfolder
  SYNGEN_DEVICE     cuda | cpu                          (default: cuda if available)
  SYNGEN_UNIFONT    path to unifont.pickle              (default data/unifont.pickle)
  SYNGEN_FORCE_DEMO set to 1 to skip model loading
"""
from __future__ import annotations

import os
import pickle
import queue
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image, ImageDraw, ImageFont

from preprocess import STYLE_H, to_png_b64

REPO_ROOT = Path(__file__).resolve().parent.parent

# character set the model was trained on (data_loader/loader.py -> letters)
LETTERS = '_Only thewigsofrcvdampbkuq.A-210xT5\'MDL,RYHJ"ISPWENj&BC93VGFKz();#:!7U64Q8?+*ZX/%'
CHARSET = "".join(sorted(set(LETTERS) - {" "}))
MAX_WORD_LEN = 24
MAX_WORDS = 60

DEFAULT_VAES = [
    str(REPO_ROOT / "model_zoo" / "stable-diffusion-v1-5"),
    "stable-diffusion-v1-5/stable-diffusion-v1-5",
    "runwayml/stable-diffusion-v1-5",
]


def _env_path(name: str, default: Path) -> Path:
    v = os.environ.get(name)
    p = Path(v) if v else default
    return p if p.is_absolute() else (REPO_ROOT / p)


class _SafeGlyphUnpickler(pickle.Unpickler):
    """unifont.pickle only holds lists/dicts of numpy arrays - refuse anything else."""
    _ALLOWED = {("numpy", "ndarray"), ("numpy", "dtype"),
                ("numpy.core.multiarray", "_reconstruct"), ("numpy._core.multiarray", "_reconstruct")}

    def find_class(self, module, name):
        if (module, name) not in self._ALLOWED:
            raise pickle.UnpicklingError(f"unifont.pickle references {module}.{name}, which is not allowed")
        return super().find_class(module, name)


def _rel(p: Path) -> str:
    try:
        return str(p.relative_to(REPO_ROOT)).replace("\\", "/")
    except ValueError:
        return str(p)


def parse_text(text: str) -> list[list[str]]:
    """Split text into lines of words and validate the character set."""
    lines = [ln.split() for ln in text.replace("\r", "").split("\n")]
    lines = [ln for ln in lines if ln]
    if not lines:
        raise ValueError("Enter some text to generate.")
    words = [w for ln in lines for w in ln]
    if len(words) > MAX_WORDS:
        raise ValueError(f"Too many words ({len(words)}). The demo limit is {MAX_WORDS}.")
    bad = sorted({c for w in words for c in w if c not in CHARSET})
    if bad:
        raise ValueError("Unsupported characters: " + " ".join(repr(c) for c in bad))
    long = [w for w in words if len(w) > MAX_WORD_LEN]
    if long:
        raise ValueError(f"Words longer than {MAX_WORD_LEN} characters are not supported: {long[0]}")
    return lines


# ---------------------------------------------------------------- composition
def trim_word(img: np.ndarray, pad: int = 3) -> np.ndarray:
    ink_cols = np.where((img < 170).any(axis=0))[0]
    if ink_cols.size == 0:
        return img
    x0, x1 = max(0, ink_cols.min() - pad), min(img.shape[1], ink_cols.max() + pad + 1)
    return img[:, x0:x1]


def compose_page(lines: list[list[np.ndarray]], gap: int = 16, line_gap: int = 6, margin: int = 20) -> np.ndarray:
    rows = []
    for words in lines:
        parts = []
        for i, w in enumerate(words):
            if i:
                parts.append(np.full((STYLE_H, gap), 255, np.uint8))
            parts.append(w)
        rows.append(np.hstack(parts))
    width = max(r.shape[1] for r in rows) + 2 * margin
    height = len(rows) * STYLE_H + (len(rows) - 1) * line_gap + 2 * margin
    page = np.full((height, width), 255, np.uint8)
    y = margin
    for r in rows:
        page[y:y + STYLE_H, margin:margin + r.shape[1]] = r
        y += STYLE_H + line_gap
    return page


# ----------------------------------------------------------------- demo font
def _find_demo_font() -> str | None:
    candidates = [
        "C:/Windows/Fonts/Inkfree.ttf", "C:/Windows/Fonts/segoepr.ttf", "C:/Windows/Fonts/segoesc.ttf",
        "/System/Library/Fonts/Supplemental/Bradley Hand Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Oblique.ttf",
    ]
    return next((c for c in candidates if os.path.exists(c)), None)


def demo_render(word: str, stroke_px: float) -> np.ndarray:
    path = _find_demo_font()
    size = 40
    font = ImageFont.truetype(path, size) if path else ImageFont.load_default()
    canvas = Image.new("L", (size * (len(word) + 2), STYLE_H), 255)
    draw = ImageDraw.Draw(canvas)
    bbox = draw.textbbox((0, 0), word, font=font)
    y = (STYLE_H - (bbox[3] - bbox[1])) // 2 - bbox[1]
    sw = 1 if stroke_px > 4.5 else 0
    draw.text((size // 2, y), word, fill=30, font=font, stroke_width=sw, stroke_fill=30)
    img = np.array(canvas)
    return trim_word(img)


# --------------------------------------------------------------------- engine
class Engine:
    def __init__(self) -> None:
        self.mode = "loading"          # loading | model | demo
        self.reason = "Loading model..."
        self.device = "cpu"
        self.vae_source = None
        self.content_source = None
        self.load_seconds = None
        self.ckpt_path = _env_path("SYNGEN_CKPT", REPO_ROOT / "model_zoo" / "One-DM-ckpt.pt")
        self.cfg_path = _env_path("SYNGEN_CFG", REPO_ROOT / "configs" / "IAM64.yml")
        self.unifont_path = _env_path("SYNGEN_UNIFONT", REPO_ROOT / "data" / "unifont.pickle")
        self._lock = threading.Lock()
        self.jobs: dict[str, dict] = {}
        self._queue: "queue.Queue[str]" = queue.Queue()
        threading.Thread(target=self._load, daemon=True).start()
        threading.Thread(target=self._worker, daemon=True).start()

    # ------------------------------------------------------------- status
    def status(self) -> dict:
        return {
            "mode": self.mode,
            "reason": self.reason,
            "device": self.device,
            "checkpoint": _rel(self.ckpt_path),
            "checkpoint_found": self.ckpt_path.exists(),
            "vae": self.vae_source,
            "content_glyphs": self.content_source,
            "load_seconds": self.load_seconds,
            "charset": CHARSET,
            "max_words": MAX_WORDS,
            "max_word_len": MAX_WORD_LEN,
            "queue": self._queue.qsize(),
        }

    # -------------------------------------------------------------- loading
    def _demo(self, reason: str) -> None:
        self.mode, self.reason = "demo", reason
        print(f"[syngen] DEMO mode: {reason}", flush=True)

    def _load(self) -> None:
        t0 = time.time()
        if os.environ.get("SYNGEN_FORCE_DEMO") == "1":
            return self._demo("SYNGEN_FORCE_DEMO=1 is set.")
        if not self.ckpt_path.exists():
            return self._demo(f"Checkpoint not found at {_rel(self.ckpt_path)}.")
        try:
            import torch
        except ImportError:
            return self._demo("PyTorch is not installed (see server/requirements-model.txt).")
        try:
            if str(REPO_ROOT) not in sys.path:
                sys.path.insert(0, str(REPO_ROOT))
            from models.unet import UNetModel
            from models.diffusion import Diffusion

            want = os.environ.get("SYNGEN_DEVICE")
            self.device = want or ("cuda" if torch.cuda.is_available() else "cpu")
            self.reason = f"Loading UNet on {self.device}..."

            with open(self.cfg_path, "r") as f:
                mcfg = yaml.safe_load(f)["MODEL"]
            unet = UNetModel(in_channels=mcfg["IN_CHANNELS"], model_channels=mcfg["EMB_DIM"],
                             out_channels=mcfg["OUT_CHANNELS"], num_res_blocks=mcfg["NUM_RES_BLOCKS"],
                             attention_resolutions=(1, 1), channel_mult=(1, 1),
                             num_heads=mcfg["NUM_HEADS"], context_dim=mcfg["EMB_DIM"])
            try:
                state = torch.load(self.ckpt_path, map_location="cpu", weights_only=True)
            except Exception:
                state = torch.load(self.ckpt_path, map_location="cpu", weights_only=False)
            if isinstance(state, dict) and "model_state_dict" in state:
                state = state["model_state_dict"]
            state = {(k[7:] if k.startswith("module.") else k): v for k, v in state.items()}
            unet.load_state_dict(state)
            unet.to(self.device).eval()
            self.unet = unet
            self.diffusion = Diffusion(device=self.device)

            self.reason = "Loading Stable Diffusion VAE (first run downloads ~330 MB)..."
            self.vae = self._load_vae()
            self.glyphs = self._load_glyphs()
            self.torch = torch
            self.load_seconds = round(time.time() - t0, 1)
            self.mode, self.reason = "model", f"One-DM UNet loaded on {self.device}."
            print(f"[syngen] model ready on {self.device} in {self.load_seconds}s", flush=True)
        except Exception as e:  # noqa: BLE001
            traceback.print_exc()
            self._demo(f"Model failed to load: {e}")

    def _load_vae(self):
        from diffusers import AutoencoderKL
        sources = [os.environ["SYNGEN_VAE"]] if os.environ.get("SYNGEN_VAE") else DEFAULT_VAES
        errors = []
        for src in sources:
            if src.startswith(("/", "\\")) or ":" in src[:3]:
                if not Path(src).exists():
                    continue
            try:
                vae = AutoencoderKL.from_pretrained(src, subfolder="vae")
                vae.to(self.device).requires_grad_(False)
                self.vae_source = src
                return vae
            except Exception as e:  # noqa: BLE001
                errors.append(f"{src}: {e}")
        raise RuntimeError("Could not load the VAE. " + " | ".join(errors))

    def _load_glyphs(self) -> dict:
        """char -> 16x16 float32 glyph (1 = ink), same as ContentData in loader.py."""
        if self.unifont_path.exists():
            with open(self.unifont_path, "rb") as f:
                symbols = _SafeGlyphUnpickler(f).load()
            symbols = {s["idx"][0]: s["mat"].astype(np.float32) for s in symbols}
            self.content_source = "unifont.pickle"
            return {c: symbols[ord(c)] for c in CHARSET}
        # fallback: rasterise each character at 16x16
        self.content_source = "rendered 16x16 fallback (data/unifont.pickle not found)"
        font = None
        for cand in ["C:/Windows/Fonts/consola.ttf", "C:/Windows/Fonts/cour.ttf",
                     "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]:
            if os.path.exists(cand):
                font = ImageFont.truetype(cand, 15)
                break
        font = font or ImageFont.load_default()
        glyphs = {}
        for c in CHARSET:
            im = Image.new("L", (16, 16), 0)
            d = ImageDraw.Draw(im)
            bb = d.textbbox((0, 0), c, font=font)
            d.text(((16 - (bb[2] - bb[0])) // 2 - bb[0], (16 - (bb[3] - bb[1])) // 2 - bb[1]), c, fill=255, font=font)
            glyphs[c] = (np.array(im) > 100).astype(np.float32)
        return glyphs

    # ----------------------------------------------------------------- jobs
    def submit(self, style: np.ndarray, laplace: np.ndarray, metrics: dict, lines: list[list[str]],
               method: str, steps: int, eta: float, seed: int) -> str:
        job_id = uuid.uuid4().hex[:12]
        total = sum(len(l) for l in lines)
        with self._lock:
            self.jobs[job_id] = {
                "id": job_id, "status": "queued", "progress": 0.0, "stage": "Waiting in queue",
                "words_done": 0, "words_total": total, "created": time.time(), "elapsed": 0.0,
                "params": {"method": method, "steps": steps, "eta": eta, "seed": seed},
                "_input": (style, laplace, metrics, lines),
            }
            # keep memory bounded
            if len(self.jobs) > 40:
                for old in sorted(self.jobs.values(), key=lambda j: j["created"])[:len(self.jobs) - 40]:
                    if old["status"] in ("done", "error"):
                        self.jobs.pop(old["id"], None)
        self._queue.put(job_id)
        return job_id

    def get_job(self, job_id: str) -> dict | None:
        job = self.jobs.get(job_id)
        if job is None:
            return None
        return {k: v for k, v in job.items() if not k.startswith("_")}

    def _update(self, job: dict, **kw) -> None:
        with self._lock:
            job.update(kw)
            job["elapsed"] = round(time.time() - job["started"], 2) if "started" in job else 0.0

    def _worker(self) -> None:
        while True:
            job_id = self._queue.get()
            job = self.jobs.get(job_id)
            if job is None:
                continue
            while self.mode == "loading":
                self._update(job, stage="Waiting for model to finish loading")
                time.sleep(0.5)
            job["started"] = time.time()
            try:
                self._run(job)
            except Exception as e:  # noqa: BLE001
                traceback.print_exc()
                self._update(job, status="error", error=str(e), stage="Failed")
            finally:
                job.pop("_input", None)

    def _run(self, job: dict) -> None:
        style, laplace, metrics, lines = job["_input"]
        p = job["params"]
        mode = self.mode
        self._update(job, status="running", stage="Starting", mode=mode)
        total = job["words_total"]
        word_imgs: list[list[np.ndarray]] = []
        words_out = []
        done = 0
        for line in lines:
            row = []
            for word in line:
                self._update(job, stage=f"Generating \u201c{word}\u201d ({done + 1}/{total})")
                if mode == "model":
                    def cb(step, n_steps, _d=done):
                        self._update(job, progress=round((_d + step / n_steps) / total, 4))
                    img = self._generate_word(word, style, laplace, p, seed_offset=done, progress=cb)
                else:
                    img = demo_render(word, metrics.get("stroke_px", 3))
                img = trim_word(img)
                row.append(img)
                words_out.append({"word": word, "image": to_png_b64(img)})
                done += 1
                self._update(job, words_done=done, progress=round(done / total, 4))
            word_imgs.append(row)
        page = compose_page(word_imgs)
        self._update(job, status="done", stage="Done", progress=1.0, words=words_out,
                     page=to_png_b64(page), page_size=[int(page.shape[1]), int(page.shape[0])])

    # ------------------------------------------------------------ sampling
    def _generate_word(self, word, style, laplace, p, seed_offset, progress) -> np.ndarray:
        torch = self.torch
        dev = self.device
        torch.manual_seed(int(p["seed"]) + seed_offset)
        style_t = torch.from_numpy(style.astype(np.float32) / 255.0)[None, None].to(dev)
        lap_t = torch.from_numpy(laplace.astype(np.float32) / 255.0)[None, None].to(dev)
        glyphs = np.stack([self.glyphs[c] for c in word])
        content = torch.from_numpy(1.0 - glyphs)[None].to(dev)              # (1, L, 16, 16)
        x = torch.randn((1, 4, style_t.shape[2] // 8, content.shape[1] * 32 // 8), device=dev)

        d = self.diffusion
        with torch.no_grad():
            if p["method"] == "ddpm":
                n_steps = d.noise_steps
                for k, i in enumerate(reversed(range(n_steps))):
                    t = torch.full((1,), i, dtype=torch.long, device=dev)
                    eps = self.unet(x, t, style_t, lap_t, content)
                    alpha, alpha_hat, beta = d.alpha[t][:, None, None, None], d.alpha_hat[t][:, None, None, None], d.beta[t][:, None, None, None]
                    noise = torch.randn_like(x) if i > 0 else torch.zeros_like(x)
                    x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / torch.sqrt(1 - alpha_hat)) * eps) + torch.sqrt(beta) * noise
                    if k % 10 == 0:
                        progress(k + 1, n_steps)
            else:  # DDIM - same schedule as Diffusion.ddim_sample
                times = torch.linspace(-1, d.noise_steps - 1, steps=p["steps"] + 1).int().tolist()
                pairs = list(zip(reversed(times[1:]), reversed(times[:-1])))
                for k, (tc, tn) in enumerate(pairs):
                    t = torch.full((1,), tc, dtype=torch.long, device=dev)
                    eps = self.unet(x, t, style_t, lap_t, content)
                    beta = d.beta[t][:, None, None, None]
                    ah = d.alpha_hat[t][:, None, None, None]
                    x0 = (x - (1 - ah).sqrt() * eps) / ah.sqrt()
                    if tn < 0:
                        x = x0
                    else:
                        ahn = d.alpha_hat[torch.full((1,), tn, dtype=torch.long, device=dev)][:, None, None, None]
                        sigma = p["eta"] * (beta * (1 - ahn) / (1 - ah)).sqrt()
                        c = (1 - ahn - sigma ** 2).sqrt()
                        x = x0 * ahn.sqrt() + c * eps + sigma * torch.randn_like(x)
                    progress(k + 1, len(pairs))
            img = self.vae.decode(x / 0.18215).sample
        img = (img / 2 + 0.5).clamp(0, 1)[0].mean(0).cpu().numpy()
        return (img * 255).round().astype(np.uint8)
