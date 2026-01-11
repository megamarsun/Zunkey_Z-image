# Zunkey_Z-image v1.0.0
# Z-Image Turbo batch GUI for 1-line-per-situation prompts
#
# Features:
# - Left: prompts (1 line = 1 situation), common style prompt, output folder, prefix.
# - Right: image preview, per-image full_prompt display, navigation (←/→), log.
# - Batch run saves:
#     * PNG images as {prefix}_0000.png, {prefix}_0001.png, ...
#     * Per-image .txt with full_prompt and metadata.
# - prompts.txt + prompts_backup/ (timestamped) auto backup.
# - "Preview: load folder" button to inspect an existing batch folder.
# - Output folder has "Open" button to open in Explorer/Finder.
# - Keyboard Left/Right keys navigate preview (ignored while editing text fields).
# - "Details" foldable section for rarely changed parameters:
#     * width / height / steps / guidance_scale / base_seed / model_id
#
# Usage:
#   python Zunkey_Z_image_v1_0_0.py

import os
import sys
import time
import json
import logging
from pathlib import Path
import threading
import traceback
import subprocess

import tkinter as tk
from tkinter import ttk, messagebox, filedialog

from PIL import Image, ImageTk

import torch
from diffusers import DiffusionPipeline

APP_NAME = "Zunkey_Z-image"
APP_VERSION = "1.0.0"
WINDOW_TITLE = f"{APP_NAME} v{APP_VERSION}"

# Default model: Z-Image Turbo text-to-image
MODEL_ID_DEFAULT = "Tongyi-MAI/Z-Image-Turbo"


# ---------------------------------------------------------------------------
# Helpers: paths, cache, config, logging
# ---------------------------------------------------------------------------

def get_script_dir() -> Path:
    return Path(__file__).resolve().parent


def get_config_path() -> Path:
    return get_script_dir() / "zunkey_z_image_config.json"


def setup_cache() -> Path:
    """
    Create ./tmp for all heavy caches + log so the app stays self-contained.
    """
    script_dir = get_script_dir()
    cache_root = script_dir / "tmp"
    cache_root.mkdir(parents=True, exist_ok=True)

    # Hugging Face / torch caches
    for name in (
        "HF_HOME",
        "HUGGINGFACE_HUB_CACHE",
        "TRANSFORMERS_CACHE",
        "DIFFUSERS_CACHE",
        "TORCH_HOME",
    ):
        os.environ.setdefault(name, str(cache_root))

    # Simple log file
    log_path = cache_root / "zunkey_z_image_log.txt"
    try:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_path, encoding="utf-8")],
        )
        logging.info("%s %s cache initialized at %s", APP_NAME, APP_VERSION, cache_root)
    except Exception:
        # logging is best-effort; don't crash if it fails
        pass

    return cache_root


def is_mostly_black(img, threshold=5.0):
    """
    Rough heuristic to detect "all black" images (e.g., NaN / failure output).
    """
    import numpy as np

    arr = np.array(img.convert("L"), dtype="float32")
    mean = float(arr.mean())
    std = float(arr.std())
    return (mean < threshold) and (std < threshold)


def load_config() -> dict:
    path = get_config_path()
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_config(data: dict) -> None:
    path = get_config_path()
    try:
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to save config: %s", e)


def backup_prompts(text: str) -> None:
    """
    Save a timestamped backup of the raw prompts text to ./prompts_backup.
    """
    script_dir = get_script_dir()
    backup_dir = script_dir / "prompts_backup"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    backup_path = backup_dir / f"prompts_{ts}.txt"
    try:
        backup_path.write_text(text, encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to backup prompts: %s", e)


def save_prompts_file(text: str) -> Path:
    """
    Save prompts.txt beside the script, and also create a timestamped backup.
    """
    script_dir = get_script_dir()
    prompts_path = script_dir / "prompts.txt"
    backup_prompts(text)
    try:
        prompts_path.write_text(text, encoding="utf-8")
    except Exception as e:
        logging.getLogger(__name__).warning("Failed to save prompts.txt: %s", e)
    return prompts_path


# ---------------------------------------------------------------------------
# Z-Image Engine wrapper
# ---------------------------------------------------------------------------

class ZImageEngine:
    """
    Small wrapper around DiffusionPipeline (Z-Image Turbo).
    Keeps the pipeline loaded between calls.
    """

    def __init__(self, model_id: str = MODEL_ID_DEFAULT):
        self.model_id = model_id
        self.cache_dir = setup_cache()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.pipe = None
        self.dtype = None
        self.lock = threading.Lock()
        self.logger = logging.getLogger(self.__class__.__name__)

    def _pick_dtype(self):
        if not torch.cuda.is_available():
            return torch.float32
        if torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float16

    def _load_pipeline(self):
        if self.pipe is not None:
            return

        # Prefer dedicated ZImagePipeline if available, otherwise trust_remote_code.
        try:
            from diffusers import ZImagePipeline  # type: ignore
            PipeCls = ZImagePipeline
            use_zimage = True
        except Exception:
            PipeCls = DiffusionPipeline
            use_zimage = False

        self.dtype = self._pick_dtype()

        if use_zimage:
            pipe = PipeCls.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                cache_dir=str(self.cache_dir),
            )
        else:
            pipe = DiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=self.dtype,
                cache_dir=str(self.cache_dir),
                trust_remote_code=True,
            )

        pipe.to(self.device)

        try:
            pipe.set_progress_bar_config(disable=True)
        except Exception:
            pass

        # VAE tiling: automatically enable when VRAM is small, but allow override via env.
        tiling_env = os.environ.get("Z_IMAGE_VAE_TILING", "").strip().lower()
        enable_tiling = None
        if tiling_env in ("1", "true", "yes", "on"):
            enable_tiling = True
        elif tiling_env in ("0", "false", "no", "off"):
            enable_tiling = False

        if enable_tiling is None:
            try:
                total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
                enable_tiling = total < 18.0
            except Exception:
                enable_tiling = False

        if enable_tiling:
            try:
                pipe.enable_vae_tiling()
            except Exception:
                pass

        self.pipe = pipe
        self.logger.info(
            "Pipeline loaded: model_id=%s, dtype=%s, device=%s",
            self.model_id,
            self.dtype,
            self.device,
        )

    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 9,
        guidance_scale: float = 0.0,
        num_images_per_prompt: int = 1,
        seed: int | None = None,
        output_type: str = "pil",
        model_id: str | None = None,
    ):
        """
        Main text-to-image call to Z-Image.
        Returns (images, info).
        """
        from inspect import signature, Parameter

        with self.lock:
            if model_id is not None and model_id != self.model_id:
                self.logger.info("Model changed: %s -> %s", self.model_id, model_id)
                self.model_id = model_id
                self.pipe = None

            self._load_pipeline()
            pipe = self.pipe

        if seed is None:
            seed = int(time.time()) & 0xFFFFFFFF

        gen = torch.Generator(device=self.device).manual_seed(int(seed) & 0xFFFFFFFF)

        width = int(width)
        height = int(height)
        steps = int(steps)
        num_images_per_prompt = int(num_images_per_prompt)
        guidance_scale = float(guidance_scale)

        self.logger.info(
            "Generate start: model_id=%s, size=%dx%d, steps=%d, guidance=%.3f, num_images=%d, seed=%s",
            self.model_id,
            width,
            height,
            steps,
            guidance_scale,
            num_images_per_prompt,
            str(seed),
        )

        base_kwargs = dict(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images_per_prompt,
            generator=gen,
            output_type=output_type,
        )

        sig = signature(pipe.__call__)
        params = sig.parameters
        accepts_var_kw = any(p.kind == Parameter.VAR_KEYWORD for p in params.values())

        kw = dict(base_kwargs)

        if not accepts_var_kw:
            for k in list(kw.keys()):
                if k not in params:
                    kw.pop(k, None)

        # Some pipelines don't expose guidance_scale directly.
        if ("guidance_scale" in kw) and ("guidance_scale" not in params) and (not accepts_var_kw):
            kw.pop("guidance_scale", None)

        t0 = time.perf_counter()
        with torch.inference_mode():
            out = pipe(**kw)
        dt = time.perf_counter() - t0

        images = getattr(out, "images", None)
        if images is None:
            images = [out]

        black_flags = [is_mostly_black(img) for img in images]
        black_count = sum(1 for f in black_flags if f)

        info = {
            "time_sec": dt,
            "seed": seed,
            "num_images": len(images),
            "has_black": any(black_flags),
            "black_count": black_count,
        }

        self.logger.info(
            "Generate done: time=%.2fs, images=%d, black=%d",
            dt,
            len(images),
            black_count,
        )

        return images, info


# ---------------------------------------------------------------------------
# Tkinter GUI
# ---------------------------------------------------------------------------

class BatchGenerateApp(ttk.Frame):
    """
    Main GUI frame.
    - Left: prompts, common prompt, output folder + prefix, basic buttons.
    - Right: preview, prompt view, log.
    - Bottom: collapsible "Details" with rarely changed settings.
    """

    PRESET_SIZES = ("768", "1024", "1152", "1536")
    PRESET_STEPS = ("6", "8", "9", "12")

    def __init__(self, master, engine: ZImageEngine, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        self.engine = engine
        self.running = False

        # For preview navigation: [{index, image, image_path, prompt, txt_path}, ...]
        self.items = []
        self.current_idx = -1
        self.preview_photo = None  # keep reference to PhotoImage

        self.details_visible = False  # for collapsing the details frame

        self._build_ui()
        self._load_config_into_ui()

    # ---- UI building ----

    def _build_ui(self):
        root_frame = ttk.Frame(self)
        root_frame.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

        # Split: left (main controls) / right (preview + log)
        left = ttk.Frame(root_frame)
        left.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        right = ttk.Frame(root_frame)
        right.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        # --- Left top: prompts + common prompt ---

        ttk.Label(left, text="1行1シチュエーション（ここにコピペ）:").pack(anchor=tk.W)
        self.prompts_text = tk.Text(left, width=50, height=12)
        self.prompts_text.pack(fill=tk.BOTH, expand=True)

        script_dir = get_script_dir()
        self.prompts_path_label = ttk.Label(
            left,
            text=f"prompts.txt 保存先: {script_dir / 'prompts.txt'}",
            foreground="gray",
        )
        self.prompts_path_label.pack(anchor=tk.W, pady=(2, 6))

        ttk.Label(left, text="共通プロンプト（スタイル等 / 例: lineart, anime style ...）:").pack(anchor=tk.W)
        self.common_prompt_text = tk.Text(left, width=50, height=3)
        self.common_prompt_text.pack(fill=tk.X)
        self.common_prompt_text.insert(
            tk.END,
            "lineart, anime style, high quality, detailed, clean lines",
        )

        # --- Left middle: output folder + prefix ---

        folder_frame = ttk.LabelFrame(left, text="出力設定")
        folder_frame.pack(fill=tk.X, pady=6)

        # Output folder
        row_out_dir = ttk.Frame(folder_frame)
        row_out_dir.pack(fill=tk.X, pady=2)
        ttk.Label(row_out_dir, text="保存フォルダ", width=12).pack(side=tk.LEFT)
        default_out_dir = str(script_dir / "output_batch")
        self.var_output_dir = tk.StringVar(value=default_out_dir)
        out_entry = ttk.Entry(row_out_dir, textvariable=self.var_output_dir, width=35)
        out_entry.pack(side=tk.LEFT, padx=(0, 4))
        ttk.Button(row_out_dir, text="参照", command=self._select_output_dir).pack(side=tk.LEFT)
        ttk.Button(row_out_dir, text="開く", command=self._open_output_dir).pack(side=tk.LEFT, padx=(4, 0))

        # Output prefix
        row_prefix = ttk.Frame(folder_frame)
        row_prefix.pack(fill=tk.X, pady=2)
        ttk.Label(row_prefix, text="prefix", width=12).pack(side=tk.LEFT)
        self.var_output_prefix = tk.StringVar(value="batch")
        ttk.Entry(row_prefix, textvariable=self.var_output_prefix, width=20).pack(side=tk.LEFT)

        # --- Left bottom: main buttons + status ---

        btn_frame = ttk.Frame(left)
        btn_frame.pack(fill=tk.X, pady=6)

        self.btn_run = ttk.Button(btn_frame, text="バッチ開始", command=self.on_run_batch)
        self.btn_run.pack(side=tk.LEFT)

        ttk.Button(
            btn_frame,
            text="プレビュー用にフォルダ読み込み",
            command=self.on_load_folder_for_preview,
        ).pack(side=tk.LEFT, padx=6)

        self.status_var = tk.StringVar(value="待機中")
        ttk.Label(left, textvariable=self.status_var, foreground="blue").pack(anchor=tk.W, pady=2)

        # --- Details (collapsible) ---

        self.details_toggle_btn = ttk.Button(
            left,
            text="▶ 詳細設定（普段はそのままでOK）",
            command=self._toggle_details,
        )
        self.details_toggle_btn.pack(fill=tk.X, pady=(4, 0))

        self.details_frame = ttk.Frame(left)
        # initially hidden; _toggle_details will pack it when needed

        self._build_details_frame(self.details_frame)

        # --- Right: preview + prompt + log ---

        preview_frame = ttk.Frame(right)
        preview_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(preview_frame, text="プレビュー画像：").pack(anchor=tk.W)
        self.preview_label = ttk.Label(preview_frame)
        self.preview_label.pack(fill=tk.BOTH, expand=False, pady=(0, 4))

        # Navigation buttons
        nav_frame = ttk.Frame(preview_frame)
        nav_frame.pack(fill=tk.X, pady=2)
        self.btn_prev = ttk.Button(nav_frame, text="← 前", command=self.on_prev, state=tk.DISABLED)
        self.btn_prev.pack(side=tk.LEFT)
        self.preview_index_var = tk.StringVar(value="- / -")
        ttk.Label(nav_frame, textvariable=self.preview_index_var).pack(side=tk.LEFT, padx=6)
        self.btn_next = ttk.Button(nav_frame, text="次 →", command=self.on_next, state=tk.DISABLED)
        self.btn_next.pack(side=tk.LEFT)

        # Prompt view
        ttk.Label(preview_frame, text="この画像の full_prompt：").pack(anchor=tk.W, pady=(6, 0))
        self.prompt_preview_text = tk.Text(preview_frame, height=5)
        self.prompt_preview_text.pack(fill=tk.X, expand=False)
        self.prompt_preview_text.configure(state=tk.DISABLED)

        # Log
        ttk.Label(preview_frame, text="ログ：").pack(anchor=tk.W, pady=(6, 0))
        self.log_text = tk.Text(preview_frame, height=10)
        self.log_text.pack(fill=tk.BOTH, expand=True)
        self.log_text.configure(state=tk.DISABLED)

    def _build_details_frame(self, parent: ttk.Frame):
        """
        Build contents of the "詳細設定" collapsible area.
        These are rarely changed, but useful for advanced users.
        """
        help_label = ttk.Label(
            parent,
            text="※ 解像度やステップなどを変えたいときだけ使います。",
            foreground="gray",
        )
        help_label.pack(anchor=tk.W, pady=(2, 4))

        # grid layout for neat alignment
        grid = ttk.Frame(parent)
        grid.pack(fill=tk.X)

        def add_combo(row, label, values, var_name, default):
            ttk.Label(grid, text=label, width=16).grid(row=row, column=0, sticky=tk.W, pady=1)
            var = tk.StringVar(value=str(default))
            combo = ttk.Combobox(
                grid,
                textvariable=var,
                values=values,
                width=10,
                state="normal",  # editable
            )
            combo.grid(row=row, column=1, sticky=tk.W, pady=1)
            setattr(self, var_name, var)

        def add_entry(row, label, var_name, default, width=10, hint: str | None = None):
            ttk.Label(grid, text=label, width=16).grid(row=row, column=0, sticky=tk.W, pady=1)
            var = tk.StringVar(value=str(default))
            entry = ttk.Entry(grid, textvariable=var, width=width)
            entry.grid(row=row, column=1, sticky=tk.W, pady=1)
            setattr(self, var_name, var)
            if hint:
                ttk.Label(grid, text=hint, foreground="gray").grid(
                    row=row, column=2, sticky=tk.W, pady=1
                )

        add_combo(0, "width", self.PRESET_SIZES, "var_width", "1024")
        add_combo(1, "height", self.PRESET_SIZES, "var_height", "1024")
        add_combo(2, "steps", self.PRESET_STEPS, "var_steps", "9")

        add_entry(
            3,
            "guidance_scale",
            "var_guidance",
            "0.0",
            hint="通常は 0.0 のままでOK",
        )

        add_entry(
            4,
            "base_seed(-1=rand)",
            "var_seed_base",
            "-1",
            hint="-1: 毎回ランダム / 0以上: 行番号を足して固定",
        )

        add_entry(
            5,
            "model_id",
            "var_model_id",
            MODEL_ID_DEFAULT,
            width=32,
            hint="通常は変更不要（上級者向け）",
        )

    # ---- Details show/hide ----

    def _toggle_details(self):
        if self.details_visible:
            self.details_frame.pack_forget()
            self.details_visible = False
            self.details_toggle_btn.configure(text="▶ 詳細設定（普段はそのままでOK）")
        else:
            self.details_frame.pack(fill=tk.X, pady=(0, 4))
            self.details_visible = True
            self.details_toggle_btn.configure(text="▼ 詳細設定（上級者向け）")

    # ---- Config load/save ----

    def _load_config_into_ui(self):
        cfg = load_config()
        if not cfg:
            return

        if "prompts_text" in cfg:
            self.prompts_text.delete("1.0", tk.END)
            self.prompts_text.insert(tk.END, cfg["prompts_text"])

        if "common_prompt" in cfg:
            self.common_prompt_text.delete("1.0", tk.END)
            self.common_prompt_text.insert(tk.END, cfg["common_prompt"])

        if "output_dir" in cfg:
            self.var_output_dir.set(cfg["output_dir"])
        if "output_prefix" in cfg:
            self.var_output_prefix.set(cfg["output_prefix"])

        if "width" in cfg:
            self.var_width.set(str(cfg["width"]))
        if "height" in cfg:
            self.var_height.set(str(cfg["height"]))
        if "steps" in cfg:
            self.var_steps.set(str(cfg["steps"]))
        if "guidance_scale" in cfg:
            self.var_guidance.set(str(cfg["guidance_scale"]))
        if "seed_base" in cfg:
            self.var_seed_base.set(str(cfg["seed_base"]))
        if "model_id" in cfg:
            self.var_model_id.set(cfg["model_id"])

    def _collect_config(self) -> dict:
        cfg = {
            "prompts_text": self.prompts_text.get("1.0", tk.END).strip(),
            "common_prompt": self.common_prompt_text.get("1.0", tk.END).strip(),
            "output_dir": self.var_output_dir.get(),
            "output_prefix": self.var_output_prefix.get(),
            "width": self.var_width.get(),
            "height": self.var_height.get(),
            "steps": self.var_steps.get(),
            "guidance_scale": self.var_guidance.get(),
            "seed_base": self.var_seed_base.get(),
            "model_id": self.var_model_id.get(),
        }
        return cfg

    # ---- Small helpers ----

    def _select_output_dir(self):
        current = self.var_output_dir.get().strip() or str(get_script_dir() / "output_batch")
        path = filedialog.askdirectory(initialdir=current, title="保存フォルダを選択")
        if path:
            self.var_output_dir.set(path)

    def _open_output_dir(self):
        """
        保存フォルダをエクスプローラー等で開く。
        """
        path_str = self.var_output_dir.get().strip()
        if not path_str:
            messagebox.showinfo("情報", "保存フォルダが設定されていません。")
            return

        p = Path(path_str)
        if not p.exists():
            messagebox.showinfo("情報", f"フォルダが存在しません:\n{p}")
            return

        try:
            if os.name == "nt":
                os.startfile(str(p))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(p)])
            else:
                subprocess.Popen(["xdg-open", str(p)])
        except Exception as e:
            messagebox.showerror("エラー", f"フォルダを開けませんでした:\n{e}")

    def _sanitize_dimensions(self, w: int, h: int) -> tuple[int, int]:
        """
        Clamp to [64, 2048] and multiple of 8.
        """
        def clamp_dim(v: int) -> int:
            v = max(64, min(2048, v))
            v = (v // 8) * 8
            return max(64, v)

        return clamp_dim(w), clamp_dim(h)

    # ---- Batch run ----

    def on_run_batch(self):
        if self.running:
            return

        prompts_raw = self.prompts_text.get("1.0", tk.END)
        lines = prompts_raw.splitlines()
        if not any(line.strip() for line in lines):
            messagebox.showwarning("警告", "シチュエーションが空です（1行1シチュエーションで入力してください）")
            return

        try:
            width = int(self.var_width.get())
            height = int(self.var_height.get())
            steps = int(self.var_steps.get())
            guidance_scale = float(self.var_guidance.get())
            seed_base = int(self.var_seed_base.get())
        except ValueError:
            messagebox.showerror("エラー", "width/height/steps/guidance_scale/base_seed の数値がおかしいです")
            return

        width, height = self._sanitize_dimensions(width, height)
        self.var_width.set(str(width))
        self.var_height.set(str(height))

        model_id = self.var_model_id.get().strip() or MODEL_ID_DEFAULT
        output_prefix = self.var_output_prefix.get().strip() or "batch"
        common_prompt = self.common_prompt_text.get("1.0", tk.END).strip()

        output_dir_str = self.var_output_dir.get().strip()
        if not output_dir_str:
            output_dir_str = str(get_script_dir() / "output_batch")
            self.var_output_dir.set(output_dir_str)

        out_dir = Path(output_dir_str)
        out_dir.mkdir(parents=True, exist_ok=True)

        prompts_path = save_prompts_file(prompts_raw)

        args = dict(
            lines=lines,
            width=width,
            height=height,
            steps=steps,
            guidance_scale=guidance_scale,
            seed_base=seed_base,
            model_id=model_id,
            output_prefix=output_prefix,
            common_prompt=common_prompt,
            output_dir=str(out_dir),
        )

        self.running = True
        self.btn_run.config(state=tk.DISABLED)
        self.status_var.set("バッチ実行中...")

        self.items = []
        self.current_idx = -1
        self._update_nav_buttons()
        self._update_preview_ui(None, "")

        self._clear_log()
        self._log(f"prompts.txt に保存しました: {prompts_path}")
        self._log(f"出力フォルダ: {out_dir}")
        self._log(f"有効行数: {sum(1 for l in lines if l.strip())} 行")

        threading.Thread(target=self._batch_thread, args=(args,), daemon=True).start()

    def _batch_thread(self, args: dict):
        try:
            lines = args["lines"]

            model_id = args["model_id"]
            width = args["width"]
            height = args["height"]
            steps = args["steps"]
            guidance_scale = args["guidance_scale"]
            seed_base = args["seed_base"]
            output_prefix = args["output_prefix"]
            common_prompt = args.get("common_prompt", "").strip()
            out_dir = Path(args["output_dir"])

            total = len(lines)
            self._log(f"{total} 行のテキストを処理します")

            for idx, raw in enumerate(lines):
                situation = raw.strip()
                if (not situation) or situation.startswith("#"):
                    self._log(f"[{idx}] 空行/コメント行のためスキップ")
                    continue

                if common_prompt and situation:
                    full_prompt = f"{common_prompt}, {situation}"
                else:
                    full_prompt = common_prompt or situation

                if seed_base >= 0:
                    seed_line = seed_base + idx
                else:
                    seed_line = None

                self._log(f"[{idx}] 生成中: {full_prompt}")
                try:
                    images, info = self.engine.generate(
                        prompt=full_prompt,
                        width=width,
                        height=height,
                        steps=steps,
                        guidance_scale=guidance_scale,
                        num_images_per_prompt=1,
                        seed=seed_line,
                        model_id=model_id,
                    )
                except Exception as e:
                    tb = traceback.format_exc()
                    logging.getLogger(__name__).error("Error in batch generate: %s\n%s", e, tb)
                    self._log(f"[{idx}] エラー: {e}")
                    continue

                img = images[0]
                out_path = out_dir / f"{output_prefix}_{idx:04d}.png"
                img.save(out_path)

                actual_seed = info.get("seed", seed_line)

                prompt_txt_path = out_dir / f"{output_prefix}_{idx:04d}.txt"
                try:
                    with prompt_txt_path.open("w", encoding="utf-8") as f:
                        f.write(full_prompt + "\n")
                        f.write(f"# situation: {situation}\n")
                        f.write(f"# common_prompt: {common_prompt}\n")
                        f.write(f"# line_index: {idx}\n")
                        f.write(
                            f"# width={width}, height={height}, "
                            f"steps={steps}, guidance_scale={guidance_scale}\n"
                        )
                        f.write(f"# seed={actual_seed}\n")
                except Exception as e:
                    logging.getLogger(__name__).warning(
                        "Failed to save prompt txt for %s: %s", out_path.name, e
                    )

                item = {
                    "index": idx,
                    "image": img,
                    "image_path": out_path,
                    "prompt": full_prompt,
                    "txt_path": prompt_txt_path,
                }
                self.items.append(item)

                self.current_idx = len(self.items) - 1
                self._update_preview_ui(img, full_prompt)
                self._update_nav_buttons()

                msg = f"[{idx}] 完了: {out_path}  time={info['time_sec']:.2f}s"
                if info.get("has_black"):
                    bc = info.get("black_count", 0)
                    if bc:
                        msg += f" / ⚠ 黒画像っぽいもの {bc}枚"
                    else:
                        msg += " / ⚠ 黒画像っぽいものあり"
                self._log(msg)

            self._log("バッチ処理が完了しました")
            self.after(0, self._on_batch_done)
        except Exception as e:
            tb = traceback.format_exc()
            logging.getLogger(__name__).error("Error in batch thread: %s\n%s", e, tb)
            self.after(0, self._on_batch_error, str(e))

    # ---- Folder preview ----

    def on_load_folder_for_preview(self):
        initial = self.var_output_dir.get().strip() or str(get_script_dir())
        folder = filedialog.askdirectory(initialdir=initial, title="プレビューするフォルダを選択")
        if not folder:
            return

        folder_path = Path(folder)
        prefix = self.var_output_prefix.get().strip() or "batch"

        png_files = sorted(folder_path.glob(f"{prefix}_*.png"))
        if not png_files:
            messagebox.showinfo("情報", f"{prefix}_*.png が見つかりませんでした")
            return

        items = []
        for png_path in png_files:
            stem = png_path.stem
            txt_path = png_path.with_suffix(".txt")
            prompt = ""
            if txt_path.exists():
                try:
                    with txt_path.open("r", encoding="utf-8") as f:
                        first_line = f.readline().rstrip("\n")
                        prompt = first_line
                except Exception:
                    prompt = ""
            try:
                img = Image.open(png_path).convert("RGB")
            except Exception:
                continue

            idx = -1
            try:
                idx_str = stem.split("_")[-1]
                idx = int(idx_str)
            except Exception:
                pass

            items.append(
                {
                    "index": idx,
                    "image": img,
                    "image_path": png_path,
                    "prompt": prompt,
                    "txt_path": txt_path if txt_path.exists() else None,
                }
            )

        if not items:
            messagebox.showinfo("情報", "読み込める画像がありませんでした")
            return

        self.items = items
        self.current_idx = 0
        first = self.items[0]
        self._update_preview_ui(first["image"], first["prompt"])
        self._update_nav_buttons()
        self._clear_log()
        self._log(f"フォルダ読み込み: {folder_path}")
        self._log(f"読み込んだ画像数: {len(self.items)}")

    # ---- Preview / nav ----

    def _update_preview_ui(self, pil_image: Image.Image | None, prompt_text: str):
        def _ui():
            if pil_image is not None:
                img = pil_image.copy()
                img.thumbnail((512, 512))
                self.preview_photo = ImageTk.PhotoImage(img)
                self.preview_label.configure(image=self.preview_photo)
            else:
                self.preview_photo = None
                self.preview_label.configure(image="")

            self.prompt_preview_text.configure(state=tk.NORMAL)
            self.prompt_preview_text.delete("1.0", tk.END)
            if prompt_text:
                self.prompt_preview_text.insert(tk.END, prompt_text)
            self.prompt_preview_text.configure(state=tk.DISABLED)

            if self.items and 0 <= self.current_idx < len(self.items):
                self.preview_index_var.set(f"{self.current_idx + 1} / {len(self.items)}")
            else:
                self.preview_index_var.set("- / -")

        self.after(0, _ui)

    def _update_nav_buttons(self):
        def _ui():
            if not self.items:
                self.btn_prev.config(state=tk.DISABLED)
                self.btn_next.config(state=tk.DISABLED)
                self.preview_index_var.set("- / -")
                return
            if self.current_idx <= 0:
                self.btn_prev.config(state=tk.DISABLED)
            else:
                self.btn_prev.config(state=tk.NORMAL)
            if self.current_idx >= len(self.items) - 1:
                self.btn_next.config(state=tk.DISABLED)
            else:
                self.btn_next.config(state=tk.NORMAL)
            self.preview_index_var.set(f"{self.current_idx + 1} / {len(self.items)}")

        self.after(0, _ui)

    def on_prev(self):
        if not self.items:
            return
        if self.current_idx <= 0:
            return
        self.current_idx -= 1
        item = self.items[self.current_idx]
        self._update_preview_ui(item["image"], item["prompt"])
        self._update_nav_buttons()

    def on_next(self):
        if not self.items:
            return
        if self.current_idx >= len(self.items) - 1:
            return
        self.current_idx += 1
        item = self.items[self.current_idx]
        self._update_preview_ui(item["image"], item["prompt"])
        self._update_nav_buttons()

    # Keyboard ← → (ignored while editing Text widgets)
    def on_prev_key(self, event):
        if isinstance(event.widget, tk.Text):
            return
        self.on_prev()

    def on_next_key(self, event):
        if isinstance(event.widget, tk.Text):
            return
        self.on_next()

    # ---- Log / status ----

    def _log(self, text: str):
        def _ui():
            self.log_text.configure(state=tk.NORMAL)
            self.log_text.insert(tk.END, text + "\n")
            self.log_text.see(tk.END)
            self.log_text.configure(state=tk.DISABLED)

        self.after(0, _ui)

    def _clear_log(self):
        self.log_text.configure(state=tk.NORMAL)
        self.log_text.delete("1.0", tk.END)
        self.log_text.configure(state=tk.DISABLED)

    def _on_batch_done(self):
        self.running = False
        self.btn_run.config(state=tk.NORMAL)
        self.status_var.set("完了")
        cfg = self._collect_config()
        save_config(cfg)

    def _on_batch_error(self, msg: str):
        self.running = False
        self.btn_run.config(state=tk.NORMAL)
        self.status_var.set("エラーが発生しました")
        messagebox.showerror("エラー", msg)
        cfg = self._collect_config()
        save_config(cfg)


# ---------------------------------------------------------------------------
# App wrapper
# ---------------------------------------------------------------------------

class ZunkeyZImageApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(WINDOW_TITLE)
        self.engine = ZImageEngine()

        self.main_frame = BatchGenerateApp(root, self.engine)
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Keyboard ← → navigation
        self.root.bind("<Left>", self.main_frame.on_prev_key)
        self.root.bind("<Right>", self.main_frame.on_next_key)

        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def on_close(self):
        cfg = self.main_frame._collect_config()
        save_config(cfg)
        self.root.destroy()


def main():
    root = tk.Tk()
    root.geometry("1120x720")
    ZunkeyZImageApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
