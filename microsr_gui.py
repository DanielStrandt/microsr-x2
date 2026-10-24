#!/usr/bin/env python3
"""MicroSR-X2 2x image-upscaling desktop GUI (Windows / tkinter).

Place this file beside microsr_model.py and microsr_infer.py, then run:
    python microsr_gui.py

Loads the exact V6 four-level INT2 model. The original CLI and model files
are not modified. Requires standard-library tkinter plus the existing
PyTorch, NumPy and Pillow dependencies.
"""
from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import time
from pathlib import Path
from tkinter import Tk, StringVar, BooleanVar, filedialog, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from microsr_infer import DEFAULT_MODEL, choose_device, save_image
from microsr_model import MicroSRX2, SUPPORTED_IMAGES

APP_DIR = Path(__file__).resolve().parent
PREVIEW_SIZE = (495, 360)
IMAGE_FILETYPES = [
    ("Supported images", "*.png *.jpg *.jpeg *.webp *.bmp *.tif *.tiff"),
    ("All files", "*.*"),
]
MODEL_FILETYPES = [
    ("MicroSR model files", "*.msr *.bin *.json"),
    ("All files", "*.*"),
]


def suggested_model() -> Path:
    """Prefer the compact model; accept a trainer export pair if not yet packed."""
    if DEFAULT_MODEL.is_file():
        return DEFAULT_MODEL
    for base in (APP_DIR / "microsr_int2", APP_DIR.parent / "microsr_int2"):
        if base.with_suffix(".bin").is_file() and base.with_suffix(".json").is_file():
            return base
    return DEFAULT_MODEL


def output_for_input(source: Path) -> Path:
    return source.with_name(source.stem + "_x2.png")


def _model_available(raw: str) -> bool:
    path = Path(raw).expanduser()
    if path.is_file():
        return path.suffix.lower() in (".msr", ".bin", ".json")
    if path.is_dir():
        return (path / "microsr_v6.msr").is_file() or (
            (path / "microsr_int2.bin").is_file() and (path / "microsr_int2.json").is_file()
        )
    if not path.suffix:
        return path.with_suffix(".bin").is_file() and path.with_suffix(".json").is_file()
    return False


def run_inference(
    source: Path,
    destination: Path,
    model_file: Path,
    device_choice: str = "auto",
    tile_size: int = 384,
) -> dict[str, object]:
    """Non-GUI inference function for testing and background execution.

    This is the exact same inference path as microsr_infer.py, including the
    saved four-level INT2 model, tiled bicubic reconstruction, alpha handling,
    EXIF rotation and lossless PNG writer. Never overwrites source images.
    """
    if not source.is_file():
        raise FileNotFoundError(f"Input image not found: {source}")
    if source.suffix.lower() not in SUPPORTED_IMAGES:
        raise ValueError(f"Unsupported input type: {source.suffix}")
    if destination.suffix.lower() != ".png":
        raise ValueError("Please save the output as .png (lossless, supports transparency)")
    if source.resolve() == destination.resolve():
        raise ValueError("Output cannot overwrite the input image")
    if tile_size < 0:
        raise ValueError("Tile size cannot be negative")
    device = choose_device(device_choice)
    if device == "cuda":
        import torch
        torch.backends.cudnn.allow_tf32 = False
    start = time.perf_counter()
    model = MicroSRX2.load(model_file, device=device)
    with Image.open(source) as image:
        original_size = image.size
        result = model.upscale_pil(image, tile_size=tile_size, preserve_alpha=True)
    output_size = result.size
    save_image(result, destination)
    return {
        "source": source,
        "destination": destination,
        "input_size": original_size,
        "output_size": output_size,
        "model": str(model_file),
        "device": device,
        "seconds": time.perf_counter() - start,
    }


def _display_image(path: Path, *, bicubic_reference: bool, zoom: str) -> ImageTk.PhotoImage:
    """Create preview in Tk's main thread, without changing the source file."""
    w_max, h_max = PREVIEW_SIZE
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image)
        # The preview is only RGB on a white backing; saved RGBA pixels retain alpha.
        if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
            rgba = image.convert("RGBA")
            rgb = Image.new("RGB", rgba.size, (255, 255, 255))
            rgb.paste(rgba, mask=rgba.getchannel("A"))
            image = rgb
        else:
            image = image.convert("RGB")

        if zoom == "100% center":
            # Input preview represents bicubic-2x LR; output preview is a real
            # 1:1 crop of 2x HR. Both use the same *final-image* coordinates.
            tw = min(w_max, image.width * (2 if bicubic_reference else 1))
            th = min(h_max, image.height * (2 if bicubic_reference else 1))
            if bicubic_reference:
                cw = max(1, (tw + 1) // 2)
                ch = max(1, (th + 1) // 2)
                x = max(0, (image.width - cw) // 2)
                y = max(0, (image.height - ch) // 2)
                image = image.crop((x, y, x+cw, y+ch)).resize((cw*2, ch*2), Image.Resampling.BICUBIC)
            else:
                x = max(0, (image.width-tw)//2)
                y = max(0, (image.height-th)//2)
                image = image.crop((x, y, x+tw, y+th))
        else:
            scale = min(w_max/image.width, h_max/image.height)
            display_size = (max(1, int(round(image.width*scale))),
                            max(1, int(round(image.height*scale))))
            if image.size != display_size:
                image = image.resize(display_size, Image.Resampling.BICUBIC)
        return ImageTk.PhotoImage(image)


class MicroSRGui:
    def __init__(self, root: Tk):
        self.root = root
        self.root.title("MicroSR-X2  |  INT2 Image Upscaler")
        self.root.geometry("1100x720")
        self.root.minsize(850, 630)
        self._events: queue.Queue[tuple[str, object]] = queue.Queue()
        self._worker: threading.Thread | None = None
        self._busy = False
        self._preview_images: dict[str, ImageTk.PhotoImage] = {}
        self._last_result: Path | None = None

        self.input_var = StringVar()
        self.output_var = StringVar()
        self.model_var = StringVar(value=str(suggested_model()))
        self.device_var = StringVar(value="auto")
        self.tile_var = StringVar(value="384")
        self.zoom_var = StringVar(value="Fit")
        self.status_var = StringVar(value="Choose a source image to begin.")
        self.info_var = StringVar(value="2x upscaling  •  396 INT2 weights  •  saved as lossless PNG")
        self._setup_style()
        self._build_ui()
        self.root.after(100, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _setup_style(self) -> None:
        style = ttk.Style(self.root)
        if "clam" in style.theme_names():
            style.theme_use("clam")
        style.configure("App.TFrame", background="#f4f6fa")
        style.configure("App.TLabel", background="#f4f6fa", foreground="#243049")
        style.configure("Title.TLabel", background="#f4f6fa", foreground="#17233d", font=("Segoe UI", 19, "bold"))
        style.configure("Muted.TLabel", background="#f4f6fa", foreground="#66758e", font=("Segoe UI", 10))
        style.configure("Section.TLabel", background="#f4f6fa", foreground="#273654", font=("Segoe UI", 10, "bold"))
        style.configure("Primary.TButton", padding=(19, 10), font=("Segoe UI", 11, "bold"))
        style.configure("Normal.TButton", padding=(12, 6), font=("Segoe UI", 10))
        style.configure("App.TEntry", padding=(7, 7))
        style.configure("App.TCombobox", padding=(5, 5))
        style.configure("Preview.TFrame", background="#e8eef6", relief="flat")
        style.configure("Preview.TLabel", background="#e8eef6", foreground="#536781", font=("Segoe UI", 10))

    def _build_ui(self) -> None:
        page = ttk.Frame(self.root, padding=(22, 18), style="App.TFrame")
        page.pack(fill="both", expand=True)
        page.columnconfigure(0, weight=1)
        page.rowconfigure(4, weight=1)

        header = ttk.Frame(page, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 15))
        ttk.Label(header, text="MicroSR-X2", style="Title.TLabel").pack(side="left")
        ttk.Label(header, text="396-weight INT2 super-resolution   ·   2x", style="Muted.TLabel").pack(side="right")

        fields = ttk.Frame(page, style="App.TFrame")
        fields.grid(row=1, column=0, sticky="ew")
        fields.columnconfigure(1, weight=1)

        ttk.Label(fields, text="Input image", style="Section.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 12), pady=5)
        self.input_entry = ttk.Entry(fields, textvariable=self.input_var, style="App.TEntry")
        self.input_entry.grid(row=0, column=1, sticky="ew", pady=5)
        self.input_entry.bind("<FocusOut>", lambda _: self._update_input_preview())
        self.input_button = ttk.Button(fields, text="Browse image…", style="Normal.TButton", command=self._browse_input)
        self.input_button.grid(row=0, column=2, padx=(10, 0), pady=5)

        ttk.Label(fields, text="Save as", style="Section.TLabel").grid(row=1, column=0, sticky="w", padx=(0, 12), pady=5)
        self.output_entry = ttk.Entry(fields, textvariable=self.output_var, style="App.TEntry")
        self.output_entry.grid(row=1, column=1, sticky="ew", pady=5)
        self.output_button = ttk.Button(fields, text="Choose output…", style="Normal.TButton", command=self._browse_output)
        self.output_button.grid(row=1, column=2, padx=(10, 0), pady=5)

        ttk.Label(fields, text="INT2 model", style="Section.TLabel").grid(row=2, column=0, sticky="w", padx=(0, 12), pady=5)
        self.model_entry = ttk.Entry(fields, textvariable=self.model_var, style="App.TEntry")
        self.model_entry.grid(row=2, column=1, sticky="ew", pady=5)
        self.model_button = ttk.Button(fields, text="Browse model…", style="Normal.TButton", command=self._browse_model)
        self.model_button.grid(row=2, column=2, padx=(10, 0), pady=5)

        options = ttk.Frame(page, style="App.TFrame")
        options.grid(row=2, column=0, sticky="ew", pady=(14, 14))
        ttk.Label(options, text="Device", style="Section.TLabel").pack(side="left", padx=(0, 9))
        self.device_box = ttk.Combobox(options, textvariable=self.device_var, state="readonly", values=("auto", "cuda", "cpu"), width=9)
        self.device_box.pack(side="left", padx=(0, 20))
        ttk.Label(options, text="LR tile", style="Section.TLabel").pack(side="left", padx=(0, 9))
        self.tile_box = ttk.Combobox(options, textvariable=self.tile_var, state="readonly", values=("128", "256", "384", "512", "0"), width=9)
        self.tile_box.pack(side="left", padx=(0, 20))
        ttk.Label(options, text="Preview", style="Section.TLabel").pack(side="left", padx=(0, 9))
        self.zoom_box = ttk.Combobox(options, textvariable=self.zoom_var, state="readonly", values=("Fit", "100% center"), width=15)
        self.zoom_box.pack(side="left")
        self.zoom_box.bind("<<ComboboxSelected>>", lambda _: self._update_previews())
        ttk.Label(options, text="0 = no tiling (more GPU memory)", style="Muted.TLabel").pack(side="right")

        actions = ttk.Frame(page, style="App.TFrame")
        actions.grid(row=3, column=0, sticky="ew", pady=(0, 13))
        self.run_button = ttk.Button(actions, text="Upscale image 2×", style="Primary.TButton", command=self._start)
        self.run_button.pack(side="left")
        self.folder_button = ttk.Button(actions, text="Open output folder", style="Normal.TButton", command=self._open_output_folder)
        self.folder_button.pack(side="left", padx=(12, 0))
        self.progress = ttk.Progressbar(actions, mode="indeterminate", length=160)
        self.progress.pack(side="right", padx=(12, 0))

        previews = ttk.Frame(page, style="App.TFrame")
        previews.grid(row=4, column=0, sticky="nsew")
        previews.columnconfigure(0, weight=1)
        previews.columnconfigure(1, weight=1)
        previews.rowconfigure(1, weight=1)
        ttk.Label(previews, text="Input  ·  bicubic comparison preview", style="Section.TLabel").grid(row=0, column=0, sticky="w", padx=(0, 7), pady=(0, 6))
        ttk.Label(previews, text="Output  ·  MicroSR INT2", style="Section.TLabel").grid(row=0, column=1, sticky="w", padx=(7, 0), pady=(0, 6))
        left = ttk.Frame(previews, style="Preview.TFrame")
        right = ttk.Frame(previews, style="Preview.TFrame")
        left.grid(row=1, column=0, sticky="nsew", padx=(0, 7))
        right.grid(row=1, column=1, sticky="nsew", padx=(7, 0))
        self.input_preview = ttk.Label(left, text="Choose an image", style="Preview.TLabel", anchor="center")
        self.input_preview.pack(fill="both", expand=True, padx=5, pady=5)
        self.result_preview = ttk.Label(right, text="The upscaled result will appear here", style="Preview.TLabel", anchor="center")
        self.result_preview.pack(fill="both", expand=True, padx=5, pady=5)

        footer = ttk.Frame(page, style="App.TFrame")
        footer.grid(row=5, column=0, sticky="ew", pady=(11, 0))
        ttk.Label(footer, textvariable=self.status_var, style="App.TLabel", wraplength=990).pack(anchor="w")
        ttk.Label(footer, textvariable=self.info_var, style="Muted.TLabel", wraplength=990).pack(anchor="w", pady=(4, 0))

        self._modifiable_controls = [self.input_entry, self.input_button, self.output_entry,
                                     self.output_button, self.model_entry, self.model_button]

    def _browse_input(self) -> None:
        initial_dir = str(Path(self.input_var.get()).parent) if self.input_var.get() else str(Path.home())
        path = filedialog.askopenfilename(title="Choose image to upscale", initialdir=initial_dir,
                                          filetypes=IMAGE_FILETYPES, parent=self.root)
        if path:
            self.input_var.set(path)
            self.output_var.set(str(output_for_input(Path(path))))
            self._last_result = None
            self._clear_result_preview()
            self._update_input_preview()
            self.status_var.set("Image selected. Click Upscale image 2× to run the exact INT2 model.")

    def _browse_output(self) -> None:
        initial = self.output_var.get()
        if not initial and self.input_var.get():
            initial = str(output_for_input(Path(self.input_var.get())))
        initial_path = Path(initial) if initial else Path.home() / "upscaled_x2.png"
        path = filedialog.asksaveasfilename(title="Save lossless 2× PNG", initialdir=str(initial_path.parent),
                                            initialfile=initial_path.name, defaultextension=".png",
                                            filetypes=[("PNG image", "*.png")], parent=self.root)
        if path:
            self.output_var.set(path)

    def _browse_model(self) -> None:
        original = Path(self.model_var.get()) if self.model_var.get() else APP_DIR
        initial = original if original.is_dir() else original.parent
        path = filedialog.askopenfilename(title="Choose MicroSR .msr or exported .bin/.json",
                                          initialdir=str(initial), filetypes=MODEL_FILETYPES, parent=self.root)
        if path:
            self.model_var.set(path)

    def _clear_result_preview(self) -> None:
        self._preview_images.pop("out", None)
        self.result_preview.configure(image="", text="The upscaled result will appear here")

    def _update_input_preview(self) -> None:
        src = Path(self.input_var.get().strip()) if self.input_var.get().strip() else None
        if src is None or not src.is_file():
            self._preview_images.pop("in", None)
            self.input_preview.configure(image="", text="Choose an image")
            return
        try:
            photo = _display_image(src, bicubic_reference=True, zoom=self.zoom_var.get())
            self._preview_images["in"] = photo
            self.input_preview.configure(image=photo, text="")
        except (OSError, ValueError) as exc:
            self.input_preview.configure(image="", text=f"Cannot preview: {exc}")

    def _update_previews(self) -> None:
        self._update_input_preview()
        if self._last_result and self._last_result.is_file():
            try:
                photo = _display_image(self._last_result, bicubic_reference=False, zoom=self.zoom_var.get())
                self._preview_images["out"] = photo
                self.result_preview.configure(image=photo, text="")
            except (OSError, ValueError) as exc:
                self.result_preview.configure(image="", text=f"Cannot preview: {exc}")

    def _set_busy(self, busy: bool) -> None:
        self._busy = busy
        self.run_button.configure(state="disabled" if busy else "normal")
        for ctl in self._modifiable_controls:
            ctl.configure(state="disabled" if busy else "normal")
        for ctl in (self.device_box, self.tile_box):
            ctl.configure(state="disabled" if busy else "readonly")
        if busy:
            self.progress.start(12)
        else:
            self.progress.stop()

    def _start(self) -> None:
        if self._busy:
            return
        raw_input = self.input_var.get().strip().strip('"')
        if not raw_input:
            messagebox.showwarning("Select an image", "Choose an input image first.", parent=self.root)
            return
        source = Path(raw_input).expanduser()
        if not source.is_file():
            messagebox.showerror("Image not found", f"Cannot find:\n{source}", parent=self.root)
            return
        output_text = self.output_var.get().strip().strip('"')
        destination = Path(output_text).expanduser() if output_text else output_for_input(source)
        self.output_var.set(str(destination))
        if destination.suffix.lower() != ".png":
            messagebox.showwarning("Choose PNG", "Save the result as a .png file to preserve all pixels and transparency.", parent=self.root)
            return
        if source.resolve() == destination.resolve():
            messagebox.showerror("Unsafe output", "The output path cannot be the input image.", parent=self.root)
            return
        if destination.is_file() and not messagebox.askyesno("Replace output?", f"Overwrite existing file?\n\n{destination}", parent=self.root):
            return
        raw_model = self.model_var.get().strip().strip('"')
        if not raw_model or not _model_available(raw_model):
            messagebox.showerror("Model not found",
                                 "Select model\\microsr_v6.msr or an exported microsr_int2.bin/.json pair.\n\n"
                                 "The model is not bundled with the GUI; use your exported V6 weights.",
                                 parent=self.root)
            return
        try:
            tile = int(self.tile_var.get())
            if tile < 0:
                raise ValueError("negative tile")
        except ValueError:
            messagebox.showerror("Tile size", "Choose a nonnegative tile size (384 recommended).", parent=self.root)
            return
        model_file = Path(raw_model).expanduser()
        device = self.device_var.get()
        try:
            choose_device(device)
        except RuntimeError as exc:
            messagebox.showerror("Device unavailable", str(exc), parent=self.root)
            return
        self._set_busy(True)
        self.status_var.set("Loading INT2 weights and upscaling image…")
        self.info_var.set(f"Input: {source.name}  •  Device: {device}  •  Tile: {tile} LR pixels")

        def job() -> None:
            try:
                result = run_inference(source, destination, model_file, device, tile)
            except Exception as exc:
                self._events.put(("error", f"{type(exc).__name__}: {exc}"))
            else:
                self._events.put(("done", result))

        self._worker = threading.Thread(target=job, name="microsr-inference", daemon=True)
        self._worker.start()

    def _drain_events(self) -> None:
        try:
            while True:
                action, value = self._events.get_nowait()
                self._set_busy(False)
                if action == "error":
                    self.status_var.set("Inference failed. See the error message for details.")
                    messagebox.showerror("MicroSR inference error", str(value), parent=self.root)
                elif action == "done":
                    result = value
                    assert isinstance(result, dict)
                    self._last_result = Path(result["destination"])
                    self._update_previews()
                    iw, ih = result["input_size"]
                    ow, oh = result["output_size"]
                    self.status_var.set(f"Saved: {self._last_result}")
                    self.info_var.set(f"{iw}×{ih} → {ow}×{oh}  •  "
                                      f"{result['seconds']:.2f} seconds  •  "
                                      f"{str(result['device']).upper()}  •  Lossless PNG")
        except queue.Empty:
            pass
        self.root.after(100, self._drain_events)

    def _open_output_folder(self) -> None:
        text = self.output_var.get().strip()
        folder = Path(text).parent if text else APP_DIR
        if not folder.is_dir():
            messagebox.showinfo("Folder not found", f"The output folder doesn't exist yet:\n{folder}", parent=self.root)
            return
        try:
            if sys.platform == "win32":
                os.startfile(str(folder))
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            else:
                subprocess.Popen(["xdg-open", str(folder)])
        except OSError as exc:
            messagebox.showerror("Cannot open folder", str(exc), parent=self.root)

    def _close(self) -> None:
        if self._busy and not messagebox.askyesno("Inference in progress", "Inference is still running. Close the window anyway?", parent=self.root):
            return
        self.root.destroy()


def main() -> None:
    root = Tk()
    MicroSRGui(root)
    root.mainloop()


if __name__ == "__main__":
    main()

