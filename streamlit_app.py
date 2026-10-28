"""GastroVision AI: ứng dụng RIA2C thật; không huấn luyện hoặc chọn lại cấu hình."""
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import io
import itertools
import json
import math
import os
import platform
import threading
import time
import warnings
import zipfile
from contextlib import nullcontext
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import numpy as np
import pandas as pd
from PIL import Image, UnidentifiedImageError

ROOT = Path(__file__).resolve().parent
MODEL_KEYS = ("xcit_small", "vit_base_b16", "maxvit_tiny", "swin_tiny")
MODEL_NAMES = ("XCiT-Small", "ViT-B/16", "MaxViT-Tiny", "Swin-Tiny")
PAIRS = tuple(itertools.combinations(range(4), 2))
TERMS = ("X", "V", "M", "S", "XV", "XM", "XS", "VM", "VS", "MS")
TARGETS = {
    "xcit_small": ("cls_attn_blocks.1.norm1", "BNC_CLS"),
    "vit_base_b16": ("blocks.11.norm1", "BNC_CLS"),
    "maxvit_tiny": ("stages.3.blocks.1.conv", "NCHW"),
    "swin_tiny": ("layers.3.blocks.1.norm1", "NHWC"),
}
MAX_FILES, MAX_FILE_BYTES, MAX_TOTAL_BYTES, MAX_PIXELS = 128, 20 * 1024**2, 160 * 1024**2, 25_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class AppError(RuntimeError):
    """Lỗi có thông báo rõ ràng cho người sử dụng."""


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def file_digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(4 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def read_manifest(path: str | Path | None = None) -> dict:
    p = Path(path or os.environ.get("RIA2C_MANIFEST", ROOT / "ria2c_app_manifest.json"))
    try:
        raw = p.read_bytes()
        cfg = json.loads(raw)
    except (OSError, ValueError) as e:
        raise AppError("Không đọc được manifest RIA2C. Cần cung cấp đúng ria2c_app_manifest.json.") from e
    if cfg.get("schema_version") != 1 or tuple(cfg.get("models", [])) != MODEL_KEYS:
        raise AppError("Manifest không đúng cấu hình RIA2C M=4 đã khóa.")
    if len(cfg.get("classes", [])) != 22 or len(set(cfg["classes"])) != 22:
        raise AppError("Class mapping phải có đúng 22 lớp khác nhau theo thứ tự đã khóa.")
    m = np.asarray(cfg.get("mobius"), dtype=np.float64)
    if m.shape != (10,) or not np.isfinite(m).all() or abs(m.sum() - 1) > 1e-8:
        raise AppError("Capacity không đạt normalization hoặc thiếu hệ số Möbius.")
    margins = []
    for i in range(4):
        others = [j for j in range(4) if j != i]
        for mask in range(8):
            ss = {j for k, j in enumerate(others) if mask & (1 << k)}
            margins.append(m[i] + sum(m[4 + q] for q, (a, b) in enumerate(PAIRS)
                                      if (a == i and b in ss) or (b == i and a in ss)))
    if min(margins) < -1e-8:
        raise AppError("Capacity vi phạm ít nhất một trong 32 ràng buộc monotonicity.")
    cps = cfg.get("checkpoints", [])
    if len(cps) != 20 or {(c["model"], c["fold"]) for c in cps} != {(m, k) for m in MODEL_KEYS for k in range(5)}:
        raise AppError("Cần đúng 20 checkpoint: bốn backbone, mỗi backbone năm fold.")
    for c in cps:
        if len(c.get("sha256", "")) != 64 or Path(c["filename"]).name != c["filename"]:
            raise AppError("Checkpoint manifest thiếu hash hoặc tên file không hợp lệ.")
    pre = cfg.get("preprocessing", {})
    if pre.get("resize_short_edge") != 256 or pre.get("center_crop") != 224 or pre.get("mean") != [.485, .456, .406] or pre.get("std") != [.229, .224, .225]:
        raise AppError("Tiền xử lý không khớp pipeline đã kiểm chứng.")
    cfg["fingerprint"] = digest(raw)
    cfg["minimum_monotonicity_margin"] = float(min(margins))
    return cfg


def choquet(probabilities: np.ndarray, m: np.ndarray) -> np.ndarray:
    p = np.asarray(probabilities, dtype=np.float64)
    if p.ndim != 3 or p.shape[1:] != (4, 22) or not np.isfinite(p).all():
        raise AppError("Xác suất phải có dạng N × 4 × 22 và hữu hạn.")
    if np.min(p) < -1e-7 or np.max(p) > 1 + 1e-7 or np.max(abs(p.sum(2) - 1)) > 2e-5:
        raise AppError("Đầu vào không phải phân bố softmax hợp lệ; không tự sửa hoặc chuẩn hóa lại.")
    features = np.concatenate([p.transpose(0, 2, 1), np.stack([np.minimum(p[:, i], p[:, j]) for i, j in PAIRS], axis=2)], axis=2)
    return np.einsum("nck,k->nc", features, np.asarray(m, dtype=np.float64))


def decompose(p: np.ndarray, m: np.ndarray, class_id: int) -> pd.DataFrame:
    if not 0 <= class_id < 22:
        raise AppError("ID lớp ngoài khoảng 1–22.")
    values = np.r_[p[:, class_id], [min(p[i, class_id], p[j, class_id]) for i, j in PAIRS]]
    contribution = values * m
    score = choquet(p[None], m)[0, class_id]
    if not np.isclose(contribution.sum(), score, atol=1e-12, rtol=0):
        raise AppError("Tổng phân rã Möbius không khớp điểm Choquet; dừng xuất kết quả.")
    return pd.DataFrame({"Hạng": TERMS, "Hệ số Möbius": m, "Giá trị đầu vào": values, "Đóng góp theo ảnh": contribution})


def decode_image(data: bytes) -> Image.Image:
    if not data or len(data) > MAX_FILE_BYTES:
        raise AppError("Ảnh trống hoặc vượt giới hạn 20 MB/ảnh.")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as im:
                if im.format not in ("JPEG", "PNG"):
                    raise AppError("Chỉ hỗ trợ nội dung JPG/JPEG/PNG thực sự.")
                w, h = im.size
                if min(w, h) < 32 or max(w, h) > 12000 or w * h > MAX_PIXELS:
                    raise AppError("Kích thước ảnh không phù hợp: tối thiểu 32 px, tối đa 25 triệu pixel và cạnh 12000 px.")
                im.load()
                # Giữ đúng RGB của nghiên cứu; không tự xoay EXIF, tăng màu hoặc crop bệnh lý.
                return im.convert("RGB").copy()
    except AppError:
        raise
    except (OSError, ValueError, UnidentifiedImageError, Image.DecompressionBombError, Image.DecompressionBombWarning) as e:
        raise AppError("File ảnh hỏng, không phải ảnh hợp lệ hoặc kích thước giải nén không an toàn.") from e


def collect_inputs(files: list[tuple[str, bytes]], zip_upload: tuple[str, bytes] | None = None) -> list[dict]:
    entries = list(files)
    if zip_upload:
        name, data = zip_upload
        try:
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                members = [v for v in z.infolist() if not v.is_dir()]
                if len(entries) + len(members) > MAX_FILES or sum(v.file_size for v in members) > MAX_TOTAL_BYTES:
                    raise AppError("ZIP vượt giới hạn 128 file hoặc 160 MB sau giải nén.")
                for v in members:
                    path = PurePosixPath(v.filename.replace("\\", "/"))
                    bad = path.is_absolute() or ".." in path.parts or ":" in v.filename or v.file_size > MAX_FILE_BYTES or bool(v.flag_bits & 1)
                    if bad:
                        entries.append((v.filename, b""))
                    else:
                        try:
                            entries.append((v.filename, z.read(v)))
                        except (OSError, RuntimeError, zipfile.BadZipFile):
                            entries.append((v.filename, b""))
        except (zipfile.BadZipFile, AppError) as e:
            entries.append((name, b""))
            zip_error = str(e)
        else:
            zip_error = ""
    else:
        zip_error = ""
    if len(entries) > MAX_FILES or sum(len(d) for _, d in entries) > MAX_TOTAL_BYTES:
        raise AppError("Mỗi lượt tối đa 128 file và tổng 160 MB; hãy chia thành nhiều lô.")
    items = []
    for i, (name, data) in enumerate(entries):
        key = f"{i + 1:04d}-{digest(data)[:12]}"
        error = ""
        try:
            im = decode_image(data)
            size = im.size
        except AppError as e:
            error, size = str(e), None
        if zip_error and zip_upload and name == zip_upload[0] and not data:
            error = zip_error
        items.append(dict(id=key, filename=str(name), data=data, sha256=digest(data), size=size, error=error))
    return items


def inputs_fingerprint(items: list[dict]) -> str:
    return digest(json.dumps([(v["filename"], v["sha256"]) for v in items], ensure_ascii=False).encode())


def csv_bytes(frame: pd.DataFrame) -> bytes:
    # Tránh tên file do người dùng nhập trở thành công thức khi mở CSV bằng Excel.
    safe = frame.copy()
    for col in safe.select_dtypes(include="object"):
        safe[col] = safe[col].map(lambda v: "'" + v if isinstance(v, str) and v.startswith(("=", "+", "-", "@")) else v)
    return safe.to_csv(index=False).encode("utf-8-sig")


class Engine:
    """Một khóa GPU dùng chung; không giữ mô hình/checkpoint trong cache."""
    def __init__(self, cfg: dict, model_dir: str | None = None):
        self.cfg = cfg
        self.model_dir = model_dir if model_dir is not None else os.environ.get("RIA2C_MODEL_DIR", "")
        self.lock = threading.RLock()
        self.verified: dict = {}
        self.inference_calls = 0

    def path(self, record: dict) -> Path:
        return Path(self.model_dir) / record["filename"] if self.model_dir else ROOT / record["local_relative_path"]

    def preflight(self, verify: bool = False) -> list[dict]:
        inventory = []
        for r in self.cfg["checkpoints"]:
            path = self.path(r)
            if not path.is_file():
                raise AppError(f"Thiếu checkpoint {r['model']}, fold {r['fold']}. Cần đặt đủ trọng số trong RIA2C_MODEL_DIR.")
            stat = path.stat()
            if stat.st_size != r["bytes"]:
                raise AppError(f"Dung lượng checkpoint {r['filename']} khác manifest; không nạp.")
            token = (str(path), stat.st_size, stat.st_mtime_ns)
            if verify and token not in self.verified:
                if file_digest(path) != r["sha256"]:
                    raise AppError(f"SHA-256 checkpoint {r['filename']} không khớp; không nạp.")
                self.verified[token] = True
            inventory.append(dict(model=r["model"], fold=r["fold"], bytes=stat.st_size))
        return inventory

    @staticmethod
    def libraries():
        import torch
        import timm
        from torchvision import transforms
        torch.set_num_threads(min(4, os.cpu_count() or 1))
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        tfm = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(),
                                  transforms.Normalize([.485, .456, .406], [.229, .224, .225])])
        return torch, timm, tfm

    def load_model(self, record, torch, timm, device):
        self.preflight(verify=True)
        path = self.path(record)
        # Chỉ giải tuần tự hóa checkpoint đã xác minh hash từ manifest tin cậy.
        try:
            payload = torch.load(path, map_location="cpu", weights_only=True)
        except Exception as e:
            if "Weights only load failed" not in str(e):
                raise
            payload = torch.load(path, map_location="cpu", weights_only=False)
        net = timm.create_model(record["architecture"], pretrained=False, num_classes=22, drop_path_rate=.15)
        net.load_state_dict(payload["model_state_dict"], strict=True)
        del payload
        return net.requires_grad_(False).to(device).eval()

    @staticmethod
    def cleanup(torch):
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def infer(self, items: list[dict], batch_size: int = 4, progress=None, force_cpu=False) -> tuple[np.ndarray, dict]:
        if not items or not 1 <= batch_size <= 8:
            raise AppError("Lô ảnh trống hoặc batch size ngoài khoảng 1–8.")
        torch, timm, tfm = self.libraries()
        device = "cuda" if torch.cuda.is_available() and not force_cpu else "cpu"
        with self.lock:
            self.preflight(verify=True)
            self.inference_calls += 1
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            x = torch.stack([tfm(decode_image(v["data"])) for v in items])
            probs = np.zeros((len(items), 4, 22), np.float32)
            actual_batch = batch_size
            try:
                for j, model in enumerate(MODEL_KEYS):
                    for fold in range(5):
                        record = next(r for r in self.cfg["checkpoints"] if r["model"] == model and r["fold"] == fold)
                        net = None
                        try:
                            net = self.load_model(record, torch, timm, device)
                            offset = 0
                            while offset < len(items):
                                end = min(offset + actual_batch, len(items))
                                try:
                                    with torch.inference_mode():
                                        with torch.autocast("cuda", dtype=torch.float16) if device == "cuda" else nullcontext():
                                            logits = net(x[offset:end].to(device))
                                        pp = logits.float().softmax(1).cpu().numpy()
                                    probs[offset:end, j] += pp
                                    offset = end
                                except torch.cuda.OutOfMemoryError:
                                    self.cleanup(torch)
                                    if actual_batch == 1:
                                        raise AppError("GPU không đủ bộ nhớ ngay cả batch1. Đóng tác vụ GPU khác hoặc dùng CPU.")
                                    actual_batch = 1
                            if progress:
                                progress((j * 5 + fold + 1) / 20, f"{MODEL_NAMES[j]} · fold {fold + 1}/5")
                        finally:
                            del net
                            self.cleanup(torch)
                    probs[:, j] /= np.float32(5)
            except AppError:
                raise
            except Exception as e:
                raise AppError(f"Suy luận bị dừng; chưa có kết quả RIA2C đầy đủ. {type(e).__name__}: {e}") from e
            finally:
                del x
                self.cleanup(torch)
            p = probs.astype(np.float64)
            choquet(p, np.array(self.cfg["mobius"]))
            stats = dict(seconds=time.perf_counter() - started, images=len(items), checkpoint_count=20, batch_requested=batch_size,
                         batch_actual=actual_batch, device=device, AMP=device == "cuda", torch=torch.__version__, timm=timm.__version__,
                         peak_VRAM_allocated_bytes=int(torch.cuda.max_memory_allocated()) if device == "cuda" else 0,
                         GPU=torch.cuda.get_device_name(0) if device == "cuda" else None, manifest_sha256=self.cfg["fingerprint"])
            return p, stats

    def cams(self, item: dict, p: np.ndarray, mode: str, progress=None, finite_difference=False) -> dict:
        if mode not in ("ria", "own"):
            raise AppError("Chế độ CAM không hợp lệ.")
        torch, timm, tfm = self.libraries()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        focus = int(choquet(p[None], np.array(self.cfg["mobius"]))[0].argmax())
        targets = [focus] * 4 if mode == "ria" else p.argmax(1).tolist()
        maps, audits, warnings_out, reproduced = [], [], [], []
        started = time.perf_counter()
        with self.lock:
            self.preflight(verify=True)
            x = tfm(decode_image(item["data"]))[None].to(device)
            for j, model in enumerate(MODEL_KEYS):
                raw = None
                noise = 0.0
                pp = []
                for fold in range(5):
                    record = next(r for r in self.cfg["checkpoints"] if r["model"] == model and r["fold"] == fold)
                    net = None
                    try:
                        net = self.load_model(record, torch, timm, device)
                        layer, layout = TARGETS[model]
                        values, probability, audit = branch_cam(net, net.get_submodule(layer), x, int(targets[j]), layout,
                                                               torch, finite_difference and fold == 0)
                        raw = values if raw is None else raw + values
                        noise += audit["noise_floor"]
                        pp.append(probability)
                        audits.append(dict(model=model, fold=fold, layer=layer, **audit))
                    finally:
                        del net
                        self.cleanup(torch)
                    if progress:
                        progress((j * 5 + fold + 1) / 20, f"Grad-CAM {MODEL_NAMES[j]} · fold {fold + 1}/5")
                heat = torch.nn.functional.interpolate(torch.from_numpy(np.maximum(raw, 0))[None, None].float(),
                                                        size=(224, 224), mode="bilinear", align_corners=False)[0, 0].numpy()
                span = float(heat.max() - heat.min())
                if not np.isfinite(heat).all() or span <= noise:
                    maps.append(None)
                    warnings_out.append(f"{MODEL_NAMES[j]}: Grad-CAM suy biến/không vượt ngưỡng nhiễu; không tạo heatmap thay thế.")
                else:
                    maps.append((heat - heat.min()) / span)
                reproduced.append(np.mean(pp, axis=0))
            del x
            self.cleanup(torch)
        re = np.asarray(reproduced)
        if not np.array_equal(re.argmax(1), p.argmax(1)):
            warnings_out.append("Float32 dùng cho gradient có argmax khác lần suy luận FP16; đọc mục tiêu CAM và kết quả gốc riêng.")
        return dict(maps=maps, targets=targets, mode=mode, audits=audits, warnings=warnings_out,
                    max_probability_abs_difference=float(abs(re - p).max()), seconds=time.perf_counter() - started,
                    image_sha256=item["sha256"], manifest_sha256=self.cfg["fingerprint"])


def spatial(t, layout):
    if layout == "NCHW":
        return t
    if layout == "NHWC":
        return t.permute(0, 3, 1, 2)
    patches = t[:, 1:]
    side = math.isqrt(patches.shape[1])
    if side * side != patches.shape[1]:
        raise AppError("Số token không tạo thành lưới CAM hợp lệ.")
    return patches.reshape(t.shape[0], side, side, t.shape[2]).permute(0, 3, 1, 2)


def branch_cam(net, layer, x, target, layout, torch, finite_difference=False):
    """Giữ nguyên công thức Grad-CAM xác suất đã kiểm chứng trong nghiên cứu."""
    saved = {}
    handle = layer.register_forward_hook(lambda module, args, output: saved.update(activation=output))
    try:
        with torch.enable_grad():
            xx = x.detach().requires_grad_(True)
            prob = net(xx).float().softmax(1)
            activation = saved["activation"]
            gradient = torch.autograd.grad(prob[0, target] / 5, activation)[0]
            a, g = spatial(activation.detach(), layout).double(), spatial(gradient.detach(), layout).double()
            if not torch.isfinite(a).all() or not torch.isfinite(g).all():
                raise AppError("Activation hoặc gradient không hữu hạn.")
            weights = g.mean((2, 3), keepdim=True)
            raw = (weights * a).sum(1)[0].cpu().numpy()
            floor = float((weights * a).abs().sum(1).max()) * np.finfo(np.float32).eps * 64
            audit = dict(target=target, activation_shape=list(activation.shape), gradient_max=float(g.abs().max()),
                         noise_floor=floor, parameter_gradients=sum(v.grad is not None for v in net.parameters()))
            if audit["parameter_gradients"]:
                raise AppError("Trọng số có gradient tích lũy không mong muốn.")
            probabilities = prob.detach().cpu().numpy()[0]
    finally:
        handle.remove()
    if finite_difference:
        generator = torch.Generator().manual_seed(42)
        direction = torch.randn(gradient.shape, generator=generator).to(x.device)
        if layout == "BNC_CLS":
            direction[:, 0] = 0
        analytic = float((gradient.detach() * direction).sum())
        values = []
        for sign in [1, -1]:
            h = layer.register_forward_hook(lambda module, args, output: output + sign * .01 * direction)
            try:
                with torch.no_grad():
                    values.append(float(net(x).float().softmax(1)[0, target] / 5))
            finally:
                h.remove()
        numeric = (values[0] - values[1]) / .02
        error = abs(numeric - analytic)
        if error > 2e-5 + .15 * abs(analytic):
            raise AppError("Kiểm tra sai phân gradient chưa đạt; không công bố CAM đã kiểm chứng.")
        audit.update(finite_difference_error=error, analytic=analytic, numeric=numeric)
    return raw, probabilities, audit


def predict_job(items: list[dict], engine: Engine, batch_size=4, progress=None) -> dict:
    valid = [v for v in items if not v["error"]]
    rows = {v["id"]: dict(file_id=v["id"], filename=v["filename"], status="Lỗi" if v["error"] else "Chờ", error=v["error"]) for v in items}
    p, stats = None, {}
    if valid:
        try:
            p, stats = engine.infer(valid, batch_size, progress)
            scores = choquet(p, np.array(engine.cfg["mobius"]))
            for n, v in enumerate(valid):
                order = np.argsort(-scores[n], kind="stable")
                pred, second = int(order[0]), int(order[1])
                rows[v["id"]].update(status="Thành công", class_id=pred + 1, class_name=engine.cfg["classes"][pred],
                                      Choquet_score=float(scores[n, pred]), runner_up_id=second + 1,
                                      runner_up_name=engine.cfg["classes"][second], margin=float(scores[n, pred] - scores[n, second]))
                for j, key in enumerate(MODEL_KEYS):
                    q = int(p[n, j].argmax())
                    rows[v["id"]][key] = q + 1
                    rows[v["id"]][key + "_probability"] = float(p[n, j, q])
        except (AppError, OSError, RuntimeError) as e:
            for v in valid:
                rows[v["id"]].update(status="Lỗi", error=str(e))
            p = None
    return dict(items=items, rows=list(rows.values()), valid_ids=[v["id"] for v in valid] if p is not None else [],
                probabilities=p, stats=stats, input_fingerprint=inputs_fingerprint(items), manifest=engine.cfg["fingerprint"],
                created_at=datetime.now(timezone.utc).isoformat())


def distribution(job: dict, classes: list[str]) -> pd.DataFrame:
    counts = np.zeros(22, int)
    for r in job["rows"]:
        if r["status"] == "Thành công":
            counts[int(r["class_id"]) - 1] += 1
    return pd.DataFrame({"Class ID": np.arange(1, 23), "Lớp": classes, "Số ảnh dự đoán": counts})


def ground_truth_table(data: bytes, job: dict) -> pd.DataFrame:
    try:
        frame = pd.read_csv(io.BytesIO(data), dtype=str, encoding="utf-8-sig", keep_default_na=False)
    except Exception as e:
        raise AppError("CSV nhãn không đọc được; dùng UTF-8 với filename,class_id hoặc file_id,class_id.") from e
    key = "file_id" if "file_id" in frame.columns else "filename"
    if key not in frame or "class_id" not in frame or frame[key].duplicated().any():
        raise AppError("CSV cần file_id hoặc filename không trùng, cùng cột class_id (1–22).")
    source = pd.DataFrame(job["rows"])
    if key == "filename" and source.filename.duplicated().any():
        raise AppError("Có file trùng tên: hãy dùng file_id từ CSV kết quả để gán nhãn chính xác.")
    if not set(frame[key]).issubset(set(source[key])):
        raise AppError("CSV có file không nằm trong lô đang xem.")
    if not frame.class_id.map(lambda s: s.isdecimal() and 1 <= int(s) <= 22).all():
        raise AppError("class_id phải là số nguyên hiển thị 1–22; không suy nhãn từ tên ảnh.")
    truth = frame[[key, "class_id"]].rename(columns={"class_id": "ground_truth_id"})
    truth["ground_truth_id"] = truth.ground_truth_id.astype(int)
    merged = source.merge(truth, on=key, how="inner", validate="one_to_one")
    merged = merged[merged.status == "Thành công"].copy()
    if merged.empty:
        raise AppError("Chưa có ảnh suy luận thành công và nhãn hợp lệ để tính Accuracy.")
    merged["Đúng"] = merged.class_id.astype(int) == merged.ground_truth_id
    return merged


def overlay_cam(image: Image.Image, heat: np.ndarray) -> np.ndarray:
    import matplotlib
    w, h = image.size
    rw, rh = (int(256 * w / h), 256) if w >= h else (256, int(256 * h / w))
    left, top = round((rw - 224) / 2), round((rh - 224) / 2)
    values, mask = np.zeros((rh, rw), np.float32), np.zeros((rh, rw), np.uint8)
    values[top:top + 224, left:left + 224] = heat
    mask[top:top + 224, left:left + 224] = 255
    values = np.asarray(Image.fromarray(values).resize(image.size, Image.Resampling.BILINEAR))
    mask = np.asarray(Image.fromarray(mask).resize(image.size, Image.Resampling.NEAREST)) > 0
    rgb = np.array(image)
    result = rgb.copy()
    colour = matplotlib.colormaps["turbo"](values)[..., :3] * 255
    alpha = (.65 * values[mask])[:, None]
    result[mask] = ((1 - alpha) * rgb[mask] + alpha * colour[mask]).astype(np.uint8)
    if not np.array_equal(result[~mask], rgb[~mask]):
        raise AppError("Lỗi chiếu CAM: vùng ngoài crop bị thay đổi.")
    return result


def cam_png(item: dict, result: dict) -> bytes:
    import matplotlib.pyplot as plt
    image = decode_image(item["data"])
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes = axes.flat
    axes[0].imshow(image)
    axes[0].set_title("Ảnh RGB gốc")
    for j, heat in enumerate(result["maps"]):
        if heat is not None:
            axes[j + 1].imshow(overlay_cam(image, heat))
        axes[j + 1].set_title(f"{MODEL_NAMES[j]} · target #{result['targets'][j] + 1:02d}")
        if heat is None:
            axes[j + 1].text(.5, .5, "CAM suy biến; không có heatmap", ha="center")
    axes[5].text(.05, .8, "Grad-CAM thật của backbone\nMỗi map chuẩn hóa độc lập\nKhông phải xác suất tổn thương\nKhông phải heatmap RIA2C", va="top", fontsize=12, linespacing=1.8)
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, facecolor="white")
    plt.close(fig)
    return buffer.getvalue()


def successful_selection(job: dict, file_id: str) -> tuple[dict, np.ndarray]:
    if file_id not in job["valid_ids"] or job["probabilities"] is None:
        raise AppError("Ảnh chưa có kết quả suy luận thành công.")
    item = next(v for v in job["items"] if v["id"] == file_id)
    return item, job["probabilities"][job["valid_ids"].index(file_id)]


def class_label(cid: int, cfg: dict) -> str:
    return f"#{cid + 1:02d} · {cfg['classes'][cid]}"


def result_chart(scores: np.ndarray, cfg: dict, st, key: str):
    import plotly.express as px
    top = np.argsort(-scores, kind="stable")[:5][::-1]
    frame = pd.DataFrame({"Lớp": [class_label(int(k), cfg) for k in top], "Điểm Choquet thô": scores[top]})
    fig = px.bar(frame, x="Điểm Choquet thô", y="Lớp", orientation="h", text="Điểm Choquet thô")
    fig.update_traces(marker_color="#176B87", texttemplate="%{x:.4f}", textposition="outside", cliponaxis=False)
    fig.update_layout(height=300, margin=dict(l=5, r=40, t=15, b=5), font=dict(size=14), xaxis_range=[0, max(1.0, scores.max() * 1.15)])
    st.plotly_chart(fig, use_container_width=True, key=key)


def show_result(item: dict, p: np.ndarray, cfg: dict, st, key: str):
    scores = choquet(p[None], np.array(cfg["mobius"]))[0]
    order = np.argsort(-scores, kind="stable")
    a, b = st.columns([1, 1.4], gap="large")
    with a:
        st.image(decode_image(item["data"]), caption=item["filename"], use_container_width=True)
    with b:
        st.subheader(class_label(int(order[0]), cfg))
        c1, c2 = st.columns(2)
        c1.metric("Điểm Choquet thô", f"{scores[order[0]]:.6f}")
        c2.metric("Khoảng cách hai lớp đầu", f"{scores[order[0]] - scores[order[1]]:.6f}")
        st.caption("Điểm Choquet chưa hiệu chuẩn. Không có nhãn thật thì hệ thống không kết luận đúng/sai.")
        result_chart(scores, cfg, st, key + "_scores")
    st.markdown("#### Kết quả bốn backbone")
    table = pd.DataFrame([{"Backbone": name, "Class ID": int(p[j].argmax()) + 1,
                           "Tên lớp": cfg["classes"][int(p[j].argmax())], "Xác suất lớp dự đoán": float(p[j].max())}
                          for j, name in enumerate(MODEL_NAMES)])
    st.dataframe(table, hide_index=True, use_container_width=True)


def job_download(job, cfg, st, key):
    frame = pd.DataFrame(job["rows"])
    st.download_button("Tải CSV kết quả", csv_bytes(frame), file_name="ria2c_results.csv", mime="text/csv", key=key + "csv")
    payload = dict(method="RIA2C M=4", manifest_sha256=job["manifest"], created_at=job["created_at"],
                   timing=job["stats"], rows=job["rows"], classes=cfg["classes"],
                   input_sha256=[dict(file_id=v["id"], filename=v["filename"], sha256=v["sha256"]) for v in job["items"]],
                   notice="Điểm Choquet thô, không phải xác suất hiệu chuẩn; không có ảnh gốc trong báo cáo.")
    st.download_button("Tải báo cáo JSON", json.dumps(payload, ensure_ascii=False, indent=2).encode(),
                       file_name="ria2c_report.json", mime="application/json", key=key + "json")


def choose_result(st, key):
    jobs = [(name, st.session_state.get(name)) for name in ["single_job", "batch_job"]]
    choices = []
    for name, job in jobs:
        if job:
            for v in job["items"]:
                if v["id"] in job["valid_ids"]:
                    choices.append((name, v["id"], v["filename"]))
    if not choices:
        st.info("Hãy phân loại một ảnh hoặc một lô ảnh trước; sau đó chọn ảnh tại đây, không cần tải lại.")
        return None
    selected = st.selectbox("Ảnh đã phân loại", range(len(choices)), format_func=lambda i: f"{'Ảnh đơn' if choices[i][0] == 'single_job' else 'Lô ảnh'} · {choices[i][2]} · {choices[i][1]}", key=key)
    name, file_id, _ = choices[selected]
    job = st.session_state[name]
    item, p = successful_selection(job, file_id)
    return item, p, job


def main():
    import streamlit as st
    st.set_page_config(page_title="GastroVision AI · RIA2C", page_icon="🩺", layout="wide")
    st.markdown("""<style>
    .stApp {background:#fff;color:#172B3A} .block-container {max-width:1400px;padding-top:2rem}
    h1,h2,h3 {color:#153E52} div[data-testid="stMetric"] {background:#F2F7FA;border-radius:10px;padding:14px}
    button[kind="primary"] {background:#176B87;border-color:#176B87} .stTabs [data-baseweb="tab"] {font-size:16px}
    </style>""", unsafe_allow_html=True)
    st.title("GastroVision AI")
    st.markdown("### Phân loại ảnh nội soi đường tiêu hóa 22 lớp bằng RIA2C")
    st.caption("Phục vụ nghiên cứu và minh họa khóa luận; không thay thế chẩn đoán hoặc quyết định của bác sĩ.")
    try:
        cfg = read_manifest()
    except AppError as e:
        st.error(str(e))
        st.stop()

    @st.cache_resource(show_spinner=False)
    def resource(manifest_text, model_dir):
        return Engine(json.loads(manifest_text), model_dir)

    engine = resource(json.dumps(cfg, ensure_ascii=False, sort_keys=True), os.environ.get("RIA2C_MODEL_DIR", ""))
    with st.sidebar:
        st.subheader("RIA2C · M=4")
        st.write("XCiT-Small · ViT-B/16\n\nMaxViT-Tiny · Swin-Tiny")
        batch = st.select_slider("Mini-batch", options=[1, 2, 4, 8], value=4, help="Nếu thiếu VRAM, hệ thống thử lại batch1 và không đổi mô hình.")
        st.caption("Nạp lần lượt 20 checkpoint; chỉ chạy khi bạn nhấn nút.")
        if st.button("Xóa ảnh và kết quả trong phiên", key="clear_session"):
            for key in list(st.session_state):
                if key != "clear_session":
                    del st.session_state[key]
            st.rerun()
        with st.expander("Tính toàn vẹn cấu hình"):
            st.write("Seed 42 · 5 checkpoint/fold cho mỗi backbone")
            st.write("Capacity normalization và 32 ràng buộc: đạt")
            st.caption("SHA-256 manifest: " + cfg["fingerprint"][:16] + "…")
    ready = True
    try:
        engine.preflight()
    except AppError as e:
        ready = False
        st.error(str(e))
        st.info("Không có chế độ giả lập hoặc dự đoán mặc định. Hãy cung cấp đủ trọng số đã khóa rồi khởi động lại.")
    tabs = st.tabs(["Một ảnh", "Hàng loạt", "Grad-CAM", "Cơ chế RIA2C", "Nghiên cứu & hướng dẫn"])

    def parse_uploads(files, archive=None, key="parsed"):
        raw = [(v.name, v.getvalue()) for v in files]
        z = (archive.name, archive.getvalue()) if archive else None
        signature = digest(json.dumps([(n, digest(b)) for n, b in raw] + ([(z[0], digest(z[1]))] if z else []), ensure_ascii=False).encode())
        old = st.session_state.get(key)
        if old and old["signature"] == signature:
            return old["items"]
        items = collect_inputs(raw, z)
        st.session_state[key] = dict(signature=signature, items=items)
        return items

    def run_job(items, key):
        bar = st.progress(0.0, text="Kiểm tra checkpoint và chuẩn bị ảnh…")
        with st.spinner("Đang chạy pipeline thật; lần đầu cần đọc và xác minh 20 checkpoint."):
            job = predict_job(items, engine, batch, lambda f, msg: bar.progress(f, text=msg))
        bar.empty()
        st.session_state[key] = job
        st.session_state["ui_inference_invocations"] = st.session_state.get("ui_inference_invocations", 0) + 1
        st.session_state.pop("xai_result", None)
        st.session_state.pop("xai_png", None)

    with tabs[0]:
        st.subheader("Phân loại một ảnh")
        upload = st.file_uploader("Chọn ảnh JPG, JPEG hoặc PNG", type=["jpg", "jpeg", "png"], key="single_upload")
        inputs = []
        if upload:
            try:
                inputs = parse_uploads([upload], key="parsed_single")
                if inputs[0]["error"]:
                    st.error(inputs[0]["error"])
                else:
                    st.image(decode_image(inputs[0]["data"]), caption="Ảnh đang chọn: " + upload.name, width=360)
            except AppError as e:
                st.error(str(e))
        if st.button("Phân loại ảnh", type="primary", disabled=not ready or not inputs or bool(inputs[0]["error"]), key="infer_single"):
            run_job(inputs, "single_job")
        job = st.session_state.get("single_job")
        if job:
            if inputs and inputs_fingerprint(inputs) != job["input_fingerprint"]:
                st.warning("Bạn vừa chọn ảnh mới. Kết quả dưới đây vẫn thuộc ảnh đã phân loại trước; nhấn Phân loại ảnh để cập nhật.")
            if job["manifest"] != cfg["fingerprint"]:
                st.warning("Manifest thay đổi; hãy phân loại lại để dùng cấu hình hiện tại.")
            elif job["valid_ids"]:
                item, p = successful_selection(job, job["valid_ids"][0])
                show_result(item, p, cfg, st, "single_result")
                st.caption(f"Thời gian gồm nạp checkpoint: {job['stats']['seconds']:.2f} s · {job['stats']['device'].upper()}")
                job_download(job, cfg, st, "single_")
            else:
                for row in job["rows"]:
                    st.error(row["error"])

    with tabs[1]:
        st.subheader("Phân loại 22 ảnh hoặc nhiều hơn trong một lượt")
        files = st.file_uploader("Chọn nhiều ảnh cùng lúc", type=["jpg", "jpeg", "png"], accept_multiple_files=True, key="batch_uploads")
        archive = st.file_uploader("Hoặc bổ sung một ZIP chứa ảnh", type=["zip"], key="batch_zip")
        inputs = []
        if files or archive:
            try:
                inputs = parse_uploads(files, archive, "parsed_batch")
                st.dataframe(pd.DataFrame([dict(file_id=v["id"], filename=v["filename"], size=str(v["size"]), trạng_thái=v["error"] or "Sẵn sàng") for v in inputs]), hide_index=True, use_container_width=True)
                st.caption(f"Đã nhận {len(inputs)} file; {sum(not v['error'] for v in inputs)} ảnh hợp lệ. File trùng tên có file_id riêng.")
            except AppError as e:
                st.error(str(e))
        if st.button("Phân loại hàng loạt", type="primary", disabled=not ready or not inputs or all(v["error"] for v in inputs), key="infer_batch"):
            run_job(inputs, "batch_job")
        job = st.session_state.get("batch_job")
        if job:
            if inputs and inputs_fingerprint(inputs) != job["input_fingerprint"]:
                st.warning("Danh sách tải lên đã thay đổi. Bảng dưới vẫn thuộc lô cũ cho đến khi bạn nhấn Phân loại hàng loạt.")
            if job["manifest"] != cfg["fingerprint"]:
                st.warning("Manifest thay đổi; hãy chạy lại lô ảnh trước khi dùng kết quả.")
            else:
                ok = len(job["valid_ids"])
                c1, c2, c3 = st.columns(3)
                c1.metric("Thành công", ok)
                c2.metric("Lỗi riêng từng ảnh", len(job["rows"]) - ok)
                dist = distribution(job, cfg["classes"])
                c3.metric("Số lớp thực tế dự đoán", int((dist["Số ảnh dự đoán"] > 0).sum()))
                st.dataframe(pd.DataFrame(job["rows"]), hide_index=True, use_container_width=True)
                st.caption("22 ảnh không bắt buộc tạo ra 22 nhãn khác nhau. Thống kê giữ đủ 22 lớp, kể cả lớp có số dự đoán bằng 0.")
                st.dataframe(dist, hide_index=True, use_container_width=True)
                job_download(job, cfg, st, "batch_")
                if ok:
                    gallery = st.checkbox("Xem thư viện thumbnail", key="gallery_toggle")
                    if gallery:
                        cols = st.columns(4)
                        for j, file_id in enumerate(job["valid_ids"]):
                            item, _ = successful_selection(job, file_id)
                            row = next(r for r in job["rows"] if r["file_id"] == file_id)
                            im = decode_image(item["data"])
                            im.thumbnail((300, 220))
                            cols[j % 4].image(im, caption=f"{item['filename']} · #{row['class_id']:02d} · C={row['Choquet_score']:.4f}", use_container_width=True)
                    names_by_id = {v["id"]: v["filename"] for v in job["items"]}
                    selected_id = st.selectbox("Xem chi tiết một ảnh trong lô", job["valid_ids"], format_func=lambda s, names=names_by_id: names.get(s, s) + " · " + s, key="batch_detail")
                    item, p = successful_selection(job, selected_id)
                    with st.expander("Chi tiết ảnh đang chọn"):
                        show_result(item, p, cfg, st, "batch_result")
                truth = st.file_uploader("CSV nhãn thật tùy chọn: filename,class_id hoặc file_id,class_id (ID 1–22)", type=["csv"], key="truth_upload")
                if truth:
                    try:
                        comparison = ground_truth_table(truth.getvalue(), job)
                        st.metric("Accuracy trên các ảnh được cung cấp nhãn hợp lệ", f"{comparison['Đúng'].mean():.2%}")
                        st.caption(f"Đánh giá {len(comparison)}/{ok} ảnh thành công có nhãn từ CSV; không suy nhãn qua tên file.")
                        st.dataframe(comparison, hide_index=True, use_container_width=True)
                        st.download_button("Tải bảng đối chiếu nhãn", csv_bytes(comparison), file_name="ria2c_ground_truth_comparison.csv", key="truth_download")
                    except AppError as e:
                        st.error(str(e))

    with tabs[2]:
        st.subheader("Giải thích vùng ảnh bằng Grad-CAM thật")
        selection = choose_result(st, "xai_selection")
        if selection:
            item, p, job = selection
            mode_label = st.radio("Lớp mục tiêu", ["Lớp RIA2C dự đoán", "Lớp dự đoán riêng của từng backbone"], key="xai_mode", horizontal=True)
            mode = "ria" if mode_label == "Lớp RIA2C dự đoán" else "own"
            signature = (item["sha256"], job["manifest"], mode)
            st.caption("Theo RIA2C: bốn CAM cùng giải thích lớp RIA2C chọn. Theo backbone: mỗi CAM giải thích lớp dự đoán riêng, có thể khác nhau.")
            if st.button("Tạo Grad-CAM", type="primary", disabled=not ready or job["manifest"] != cfg["fingerprint"], key="create_cam"):
                bar = st.progress(0.0, text="Chuẩn bị gradient…")
                try:
                    with st.spinner("Đang tính activation và gradient từ 20 checkpoint thật…"):
                        result = engine.cams(item, p, mode, lambda f, msg: bar.progress(f, text=msg))
                        st.session_state["xai_result"] = dict(signature=signature, result=result)
                        st.session_state["xai_png"] = cam_png(item, result)
                except (AppError, RuntimeError, OSError) as e:
                    st.error(f"Không tạo được CAM: {e}")
                finally:
                    bar.empty()
            cached = st.session_state.get("xai_result")
            if cached and cached["signature"] == signature:
                result = cached["result"]
                st.image(decode_image(item["data"]), caption="Ảnh RGB gốc", width=440)
                columns = st.columns(2)
                for j, heat in enumerate(result["maps"]):
                    with columns[j % 2]:
                        st.markdown(f"**{MODEL_NAMES[j]} · target #{result['targets'][j] + 1:02d}**")
                        if heat is None:
                            st.warning("CAM suy biến; không có heatmap hợp lệ để hiển thị.")
                        else:
                            st.image(overlay_cam(decode_image(item["data"]), heat), use_container_width=True)
                for message in result["warnings"]:
                    st.warning(message)
                st.caption("Mỗi heatmap chuẩn hóa độc lập 0–1; màu là cường độ tương đối. Vùng nóng không đồng nghĩa tổn thương đã xác nhận. Grad-CAM giải thích backbone, không trực tiếp giải thích RIA2C. Ngoài center crop giữ nguyên ảnh.")
                st.download_button("Tải hình Grad-CAM PNG", st.session_state["xai_png"], file_name="ria2c_backbone_GradCAM.png", mime="image/png", key="download_cam")
            elif cached:
                st.info("Ảnh hoặc chế độ mục tiêu đã thay đổi; nhấn Tạo Grad-CAM để tính đúng lựa chọn mới.")

    with tabs[3]:
        st.subheader("Đọc cơ chế hợp nhất RIA2C")
        selection = choose_result(st, "fusion_selection")
        if selection:
            import plotly.express as px
            item, p, job = selection
            if job["manifest"] != cfg["fingerprint"]:
                st.warning("Kết quả thuộc manifest cũ; cần phân loại lại.")
            else:
                scores = choquet(p[None], np.array(cfg["mobius"]))[0]
                predicted = int(scores.argmax())
                cid = st.selectbox("Lớp cần giải thích", list(range(22)), index=predicted, format_func=lambda k: class_label(k, cfg), key="fusion_class")
                a, b = st.columns(2)
                with a:
                    st.markdown(f"**Xác suất backbone cho lớp #{cid + 1:02d}**")
                    st.dataframe(pd.DataFrame({"Backbone": MODEL_NAMES, "Xác suất": p[:, cid]}), hide_index=True, use_container_width=True)
                with b:
                    st.markdown("**Các lớp cạnh tranh theo điểm Choquet**")
                    result_chart(scores, cfg, st, "fusion_scores")
                dec = decompose(p, np.array(cfg["mobius"]), cid)
                fig = px.bar(dec, x="Hạng", y="Đóng góp theo ảnh", text="Đóng góp theo ảnh")
                fig.update_traces(marker_color=["#176B87" if v >= 0 else "#B44141" for v in dec["Đóng góp theo ảnh"]], texttemplate="%{y:+.4f}", textposition="outside", cliponaxis=False)
                fig.update_layout(height=380, margin=dict(t=35, b=25), font=dict(size=15), title=f"Phân rã điểm lớp #{cid + 1:02d} · giữ dấu âm/dương")
                st.plotly_chart(fig, use_container_width=True, key="mobius_chart")
                st.latex(r"C_c=\sum_i m_i p_{i,c}+\sum_{i<j}m_{ij}\min(p_{i,c},p_{j,c})")
                error = abs(float(dec["Đóng góp theo ảnh"].sum()) - float(scores[cid]))
                st.success(f"Tổng 10 hạng = {dec['Đóng góp theo ảnh'].sum():.9f} · Điểm Choquet = {scores[cid]:.9f} · Sai số = {error:.2e}")
                st.caption("X: XCiT · V: ViT · M: MaxViT · S: Swin. Hệ số Möbius là tham số capacity; đóng góp theo ảnh là hệ số nhân feature. Hạng cặp dùng min của hai xác suất; không lấy trị tuyệt đối.")
                st.dataframe(dec, hide_index=True, use_container_width=True)
                st.download_button("Tải CSV phân rã Möbius", csv_bytes(dec), file_name=f"mobius_class_{cid + 1:02d}.csv", key="download_mobius")
                all_scores = pd.DataFrame({"Class ID": range(1, 23), "Tên lớp": cfg["classes"], "Điểm Choquet thô": scores})
                with st.expander("Đủ 22 điểm Choquet và Shapley của capacity"):
                    st.dataframe(all_scores, hide_index=True, use_container_width=True)
                    m = np.array(cfg["mobius"])
                    phi = [m[i] + .5 * sum(m[4 + q] for q, pair in enumerate(PAIRS) if i in pair) for i in range(4)]
                    st.dataframe(pd.DataFrame({"Backbone": MODEL_NAMES, "Shapley capacity toàn cục": phi}), hide_index=True)
                    st.caption("Shapley capacity là thuộc tính toàn cục, không phải đóng góp của riêng ảnh và không phải SHAP.")
                st.download_button("Tải đủ 22 điểm", csv_bytes(all_scores), file_name="ria2c_22_scores.csv", key="download_scores")

    with tabs[4]:
        st.subheader("GastroVision · RIA2C")
        st.write("Đề tài nghiên cứu phân loại 22 lớp ảnh nội soi đường tiêu hóa bằng hợp nhất fuzzy integral. Sáu backbone đã khảo sát gồm XCiT-Small, ViT-B/16, DeiT-Small, MaxViT-Tiny, Swin-Tiny và PVTv2-B0. Cấu hình chính RIA2C cố định bốn backbone XCiT, ViT, MaxViT và Swin.")
        st.write("RIA2C là Regularized Interaction-Aware 2-Additive Choquet. Ứng dụng dùng capacity đã học và khóa, không tối ưu khi tải ảnh. Mỗi backbone trung bình xác suất softmax của năm checkpoint, sau đó Choquet 2-additive dùng bốn hạng đơn và sáu tương tác cặp để cho điểm 22 lớp.")
        st.markdown("""**Hướng dẫn nhanh**

1. Tải ảnh đơn hoặc chọn nhiều ảnh/ZIP trong Hàng loạt, rồi nhấn nút phân loại đúng một lần.
2. Đọc lớp, điểm Choquet và xác suất từng backbone. Nhiều ảnh có thể được dự đoán cùng lớp.
3. Chọn ảnh đã xử lý trong Grad-CAM hoặc Cơ chế RIA2C; không cần tải lại. CAM chỉ tính khi nhấn nút.
4. Nếu có nhãn thật, cung cấp CSV với ID lớp 1–22. Với tên file trùng nhau, dùng file_id trong CSV kết quả.
5. Tải CSV/JSON hoặc ảnh Grad-CAM. Dùng Xóa ảnh và kết quả trong phiên khi kết thúc.
""")
        st.info("Ứng dụng không xác nhận ảnh ngoài miền là hợp lệ hay phát hiện mọi tình trạng ngoài 22 lớp. Không sử dụng để chẩn đoán hoặc đề xuất điều trị. Ảnh được xử lý trên máy/chủ ứng dụng, giữ trong bộ nhớ phiên và không tự gửi tới dịch vụ AI khác; không dùng để huấn luyện.")
        st.caption("Không hiển thị benchmark không có nguồn hoặc dùng điểm của cấu hình khác để quảng bá ứng dụng. Các hình XAI nghiên cứu đã chốt không được dùng thay heatmap của ảnh tải lên.")
        st.dataframe(pd.DataFrame({"Class ID hiển thị": range(1, 23), "Tên lớp chính thức": cfg["classes"]}), hide_index=True, use_container_width=True)


def self_test(report_dir: str):
    """Kiểm thử thật trên artifacts local; chỉ ghi vào một thư mục báo cáo mới."""
    import psutil
    from streamlit.testing.v1 import AppTest
    out = Path(report_dir).resolve()
    out.mkdir(parents=True, exist_ok=False)
    cfg = read_manifest()
    engine = Engine(cfg)
    checks = []

    def record(name, **values):
        checks.append(dict(check=name, **values))
        (out / "test_progress.json").write_text(json.dumps(checks, ensure_ascii=False, indent=2), encoding="utf-8")
        print(name, json.dumps(values, ensure_ascii=True), flush=True)

    def expect_error(function, name):
        try:
            function()
        except AppError:
            record(name, passed=True)
        else:
            raise AssertionError(name)

    # Kiểm tra lỗi phải chạy trước bài thử GPU để không bỏ sót các ranh giới đầu vào.
    expect_error(lambda: decode_image(b"not an image"), "reject_non_image")
    expect_error(lambda: decode_image(b"\x89PNG\r\n\x1a\ntruncated"), "reject_corrupt_png")
    tiny = io.BytesIO()
    Image.new("RGB", (8, 8)).save(tiny, format="PNG")
    expect_error(lambda: decode_image(tiny.getvalue()), "reject_too_small")
    badcfg = json.loads(json.dumps(cfg))
    badcfg["mobius"][0] += .1
    tempcfg = out / "invalid_manifest.json"
    tempcfg.write_text(json.dumps(badcfg), encoding="utf-8")
    expect_error(lambda: read_manifest(tempcfg), "reject_invalid_capacity")
    missing = Engine(cfg, str(out / "no_models"))
    expect_error(lambda: missing.preflight(), "reject_missing_checkpoint")

    old = ROOT / "reports/png600_delivery_20261021_192311_629150/XAI_GradCAM_22Classes_20261027_204943_1127584"
    source = json.loads((old / "selected_samples_internal.json").read_text(encoding="utf-8"))
    table = ROOT / "reports/tables/locked_official_test_v1/development_cv5_all6_v1/locked_test_metadata.csv"
    meta = pd.read_csv(table)
    base = ROOT / "artifacts/locked_official_test_v1/development_cv5_all6_v1/locked_test_evaluation"
    order = json.loads((base / "locked_test_model_order.json").read_text())["model_keys"]
    canonical = np.load(base / "locked_test_probability_tensor.npy")[:, [order.index(m) for m in MODEL_KEYS]].astype(float)
    labels = np.load(base / "locked_test_labels.npy")
    assert np.array_equal(meta.label_id, labels)
    indices = []
    special = {1: 263, 2: 273, 14: 790, 18: 1361}
    for cid in range(22):
        row = next(v for v in source if v["class_id"] == cid and v["kind"] == "main")
        ix = special.get(cid, row["sample_index"])
        if cid not in special and row["split"] != "Official Test":
            ix = int(np.where(labels == cid)[0][0])
        assert labels[ix] == cid
        indices.append(ix)
    files = [(f"class{int(meta.iloc[ix].label_id) + 1:02d}_{meta.iloc[ix].file_name}", (ROOT / meta.iloc[ix].relative_path).read_bytes()) for ix in indices]
    items = collect_inputs(files)
    assert len(items) == 22 and not any(v["error"] for v in items)
    pd.DataFrame([dict(file_id=v["id"], filename=v["filename"], sample_index=ix, ground_truth_display=int(labels[ix]) + 1, image_sha256=v["sha256"], relative_path=meta.iloc[ix].relative_path) for v, ix in zip(items, indices)]).to_csv(out / "selected_22_provenance.csv", index=False)
    original_pngs = {str(p): file_digest(p) for p in (old.parent).glob("XAI_*.png") if "backup" not in p.name}
    inputs_before = {str(p): file_digest(p) for p in [base / "locked_test_probability_tensor.npy", base / "locked_test_labels.npy", ROOT / "ria2c_app_manifest.json"]}
    sample = items[2]
    duplicates = collect_inputs([("same.png", files[0][1]), ("same.png", files[1][1])])
    assert duplicates[0]["id"] != duplicates[1]["id"]
    record("duplicate_names_have_unique_ids", passed=True)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, "w") as z:
        z.writestr("folder/valid.jpg", files[0][1])
        z.writestr("../escape.jpg", files[1][1])
        z.writestr("broken.png", b"broken")
    zipped = collect_inputs([], ("samples.zip", archive.getvalue()))
    assert len(zipped) == 3 and sum(not v["error"] for v in zipped) == 1 and not (ROOT.parent / "escape.jpg").exists()
    record("zip_valid_and_invalid_isolated_no_extraction", passed=True)

    process = psutil.Process()
    stop = threading.Event()
    peak_rss = [process.memory_info().rss]

    def memory_sampler():
        while not stop.wait(.1):
            peak_rss[0] = max(peak_rss[0], process.memory_info().rss)

    sampler = threading.Thread(target=memory_sampler, daemon=True)
    sampler.start()
    print("BEGIN_REAL_22_IMAGE_BATCH", flush=True)
    batch_job = predict_job(items + collect_inputs([("broken.png", b"broken")]), engine, 4,
                            lambda f, msg: print(f"BATCH {f:.0%} {msg}", flush=True))
    assert len(batch_job["valid_ids"]) == 22 and len(batch_job["rows"]) == 23, batch_job["rows"]
    p = batch_job["probabilities"]
    scores = choquet(p, np.array(cfg["mobius"]))
    expected = canonical[indices]
    error = float(abs(p - expected).max())
    expected_scores = choquet(expected, np.array(cfg["mobius"]))
    assert error < .005 and np.array_equal(p.argmax(2), expected.argmax(2))
    assert np.array_equal(scores.argmax(1), expected_scores.argmax(1))
    assert abs(distribution(batch_job, cfg["classes"])["Số ảnh dự đoán"].sum() - 22) == 0
    pd.DataFrame(batch_job["rows"]).to_csv(out / "batch_results_22_plus_invalid.csv", index=False)
    np.savez_compressed(out / "batch_probabilities_and_scores.npz", actual=p, canonical=expected, actual_scores=scores, canonical_scores=expected_scores)
    record("real_22_image_batch", passed=True, maximum_probability_error=error,
           maximum_Choquet_error=float(abs(scores - expected_scores).max()), backbone_predictions_match=True,
           RIA_predictions_match=True, correct_vs_GT=int((scores.argmax(1) == labels[indices]).sum()),
           distinct_predicted_classes=int((distribution(batch_job, cfg["classes"])["Số ảnh dự đoán"] > 0).sum()), **batch_job["stats"])
    single_p, single_stats = engine.infer([sample], 1)
    assert single_p.argmax(2).tolist() == expected[2:3].argmax(2).tolist()
    assert choquet(single_p, np.array(cfg["mobius"])).argmax(1).tolist() == expected_scores[2:3].argmax(1).tolist()
    record("real_single_image", passed=True, maximum_probability_error=float(abs(single_p - expected[2:3]).max()), **single_stats)
    single_job = dict(items=[sample], rows=[dict(next(v for v in batch_job["rows"] if v["file_id"] == sample["id"]))], valid_ids=[sample["id"]], probabilities=single_p,
                      stats=single_stats, input_fingerprint=inputs_fingerprint([sample]), manifest=cfg["fingerprint"], created_at=batch_job["created_at"])

    gt = pd.DataFrame({"file_id": [v["id"] for v in items], "class_id": np.arange(1, 23)})
    comparison = ground_truth_table(csv_bytes(gt), batch_job)
    assert len(comparison) == 22
    expect_error(lambda: ground_truth_table(b"filename,class_id\nunknown.jpg,2\n", batch_job), "reject_unknown_truth_filename")
    expect_error(lambda: ground_truth_table(b"filename,class_id\nsame.png,2\n", dict(rows=[dict(filename=v["filename"], file_id=v["id"], status="Lỗi") for v in duplicates])), "reject_ambiguous_truth_filename")
    record("ground_truth_mapping", passed=True, accuracy=float(comparison["Đúng"].mean()))
    for n in range(22):
        for c in range(22):
            dec = decompose(p[n], np.array(cfg["mobius"]), c)
            assert abs(dec["Đóng góp theo ảnh"].sum() - scores[n, c]) < 1e-12
    record("all_484_Mobius_decompositions", passed=True)

    # Hai chế độ XAI thật; bốn ảnh đang dùng trong phần minh họa khóa luận.
    sources = {
        2: ROOT / "reports/XAI_Class03_alternative_sample_20261028_134502_509450",
        1: ROOT / "reports/XAI_presentation_revision_20261028_143949_404738/raw_cams_verified",
        14: ROOT / "reports/XAI_presentation_revision_20261028_143949_404738/raw_cams_verified",
        18: ROOT / "reports/XAI_presentation_revision_20261028_143949_404738/raw_cams_verified",
    }
    casekeys = {1: "main_class02", 14: "error_class15", 18: "main_class19"}
    last_cam = None
    for n in [2, 1, 18, 14]:
        result = engine.cams(items[n], p[n], "own", lambda f, msg: print(f"CAM#{n + 1:02d} {f:.0%} {msg}", flush=True), True)
        differences = []
        for j, model in enumerate(MODEL_KEYS):
            path = sources[n] / (f"{model}_gradcam.npz" if n == 2 else f"{casekeys[n]}_{model}_aggregate.npz")
            oldheat = np.load(path)["cam224"]
            assert result["maps"][j] is not None
            delta = float(abs(result["maps"][j] - oldheat).max())
            assert delta < 1e-4, (n, model, delta)
            differences.append(delta)
        (out / f"GradCAM_class{n + 1:02d}.png").write_bytes(cam_png(items[n], result))
        pd.DataFrame(result["audits"]).to_csv(out / f"CAM_gradient_audit_class{n + 1:02d}.csv", index=False)
        record(f"real_GradCAM_class{n + 1:02d}_own", passed=True, map_max_errors=differences, seconds=result["seconds"], float32_probability_error=result["max_probability_abs_difference"])
        if n == 2:
            last_cam = result
    ria_cam = engine.cams(items[1], p[1], "ria", lambda f, msg: print(f"CAM_RIA_TARGET {f:.0%} {msg}", flush=True), True)
    assert ria_cam["targets"] == [int(scores[1].argmax())] * 4
    (out / "GradCAM_common_RIA_target_class02.png").write_bytes(cam_png(items[1], ria_cam))
    record("real_GradCAM_common_RIA_target", passed=True, targets=ria_cam["targets"], valid_maps=sum(v is not None for v in ria_cam["maps"]), warnings=ria_cam["warnings"])

    # AppTest dùng kết quả vừa suy luận thật; không có backend dự đoán giả.
    # AppTest dùng chung sys.argv với CLI; bỏ cờ tự kiểm thử để không gọi lồng self_test.
    original_argv = list(os.sys.argv)
    os.sys.argv = [str(Path(__file__))]
    at = AppTest.from_file(str(Path(__file__)), default_timeout=120).run()
    assert len(at.exception) == 0, at.exception
    assert len(at.tabs) == 5
    record("Streamlit_startup", passed=True, tab_count=len(at.tabs))
    at.session_state["single_job"] = single_job
    at.session_state["batch_job"] = batch_job
    at.run()
    assert len(at.exception) == 0, at.exception
    before_calls = at.session_state["ui_inference_invocations"] if "ui_inference_invocations" in at.session_state else 0
    at.checkbox(key="gallery_toggle").check().run()
    assert len(at.exception) == 0, at.exception
    at.selectbox(key="fusion_class").set_value(14).run()
    assert len(at.exception) == 0, at.exception
    at.run()
    after_calls = at.session_state["ui_inference_invocations"] if "ui_inference_invocations" in at.session_state else 0
    assert before_calls == after_calls
    assert len(at.session_state["batch_job"]["valid_ids"]) == 22
    record("Streamlit_results_gallery_fusion_reruns", passed=True, no_new_inference=True, batch_preserved=True)
    stop.set()
    sampler.join(2)
    assert all(file_digest(Path(path)) == value for path, value in {**original_pngs, **inputs_before}.items())
    record("locked_images_and_inputs_unchanged", passed=True)
    os.sys.argv = original_argv
    report = dict(checks=checks, peak_process_RSS_sampled_bytes=peak_rss[0], memory_sampling_seconds=.1,
                  no_retraining=True, no_new_capacity_fitting=True, no_Test_parameter_selection=True,
                  historical_OOF_limit="Không phải đánh giá nested toàn pipeline; đây là kiểm tra tái lập suy luận app, không phải benchmark mới.",
                  configuration_sha256=cfg["fingerprint"], python=platform.python_version(), output=str(out))
    (out / "local_verification.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("LOCAL_VERIFICATION_COMPLETE", out, flush=True)


if __name__ == "__main__":
    if "--self-test" in os.sys.argv:
        parser = argparse.ArgumentParser(description="Kiểm thử local thật, không huấn luyện.")
        parser.add_argument("--self-test", action="store_true")
        parser.add_argument("--report-dir", required=True)
        args = parser.parse_args()
        self_test(args.report_dir)
    else:
        main()
