"""Scorer adapters for direct antenna search."""

from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np
import scipy.io
import jax.numpy as jnp

from .checkpoints import forward_surrogate_from_torch_checkpoint
from .config import ScorerConfig
from .models_jax import forward_surrogate
from .pngf import (
    PNGF_FREQ_HZ,
    PNGF_TARGET_DIM,
    pack_pngf_targets,
    project_pngf_center_fed_mask_np,
)


class Scorer(Protocol):
    spectrum_dim: int

    def score(self, design: jnp.ndarray) -> jnp.ndarray:
        """Return predicted spectra shaped [batch, spectrum_dim]."""


@dataclass
class SurrogateScorer:
    params: dict
    spectrum_dim: int = 81

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "SurrogateScorer":
        return cls(forward_surrogate_from_torch_checkpoint(path))

    def score(self, design: jnp.ndarray) -> jnp.ndarray:
        return forward_surrogate(self.params, design)


@dataclass
class ExternalEMScorer:
    solver_mode: str = "air"
    command: str | None = None
    work_dir: Path | None = None
    cache_dir: Path | None = None
    timeout_seconds: float | None = None
    bad_spectrum_value: float = 1.0e6
    spectrum_dim: int = 81
    freq: np.ndarray | None = None
    matlab_function: str = "score_designs_em"

    def __post_init__(self) -> None:
        if self.solver_mode not in {"air", "substrate"}:
            raise ValueError("solver_mode must be 'air' or 'substrate'")
        self.work_dir = Path(self.work_dir or Path.cwd()).resolve()
        self.cache_dir = Path(self.cache_dir or self.work_dir / "eggroll_runs" / "em_cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = self.timeout_seconds or (900.0 if self.solver_mode == "air" else 2400.0)
        if self.freq is None:
            if self.solver_mode == "air":
                self.freq = np.linspace(10e9, 20e9, self.spectrum_dim, dtype=np.float64)
            else:
                self.freq = np.linspace(1e9, 5e9, self.spectrum_dim, dtype=np.float64)
        else:
            self.freq = np.asarray(self.freq, dtype=np.float64).reshape(-1)
            self.spectrum_dim = int(self.freq.shape[0])
        self.last_valid = np.zeros((0,), dtype=bool)
        self.last_error_message: list[str] = []

    def score(self, design: jnp.ndarray) -> jnp.ndarray:
        designs = self._normalize_designs(design)
        spectra = np.full((designs.shape[0], self.spectrum_dim), self.bad_spectrum_value, dtype=np.float32)
        valid = np.zeros((designs.shape[0],), dtype=bool)
        errors = [""] * designs.shape[0]

        misses: list[tuple[int, str, np.ndarray]] = []
        for idx, mask in enumerate(designs):
            key = self._cache_key(mask)
            cached = self._load_cache(key)
            if cached is None:
                misses.append((idx, key, mask))
            else:
                spectra[idx] = cached["spectrum"]
                valid[idx] = bool(cached["valid"])
                errors[idx] = str(cached["error"])

        if misses:
            if self.command is None:
                raise NotImplementedError(
                    "ExternalEMScorer is not implemented/configured for MATLAB/CST full-wave execution. "
                    "Provide command=... with {input_mat} and {output_mat} placeholders, or pre-populate the cache."
                )
            missed_masks = np.stack([item[2] for item in misses], axis=0)
            try:
                result = self._run_matlab(missed_masks)
            except Exception as exc:
                result = {
                    "s11_db": np.full((len(misses), self.spectrum_dim), self.bad_spectrum_value, dtype=np.float32),
                    "valid": np.zeros((len(misses),), dtype=bool),
                    "error_message": [str(exc)] * len(misses),
                }
            for local_idx, (global_idx, key, _mask) in enumerate(misses):
                row_valid = bool(result["valid"][local_idx])
                row_error = str(result["error_message"][local_idx])
                if row_valid and np.all(np.isfinite(result["s11_db"][local_idx])):
                    row_spectrum = result["s11_db"][local_idx].astype(np.float32)
                else:
                    row_spectrum = np.full((self.spectrum_dim,), self.bad_spectrum_value, dtype=np.float32)
                spectra[global_idx] = row_spectrum
                valid[global_idx] = row_valid
                errors[global_idx] = row_error
                self._save_cache(key, row_spectrum, row_valid, row_error)

        self.last_valid = valid
        self.last_error_message = errors
        return jnp.asarray(spectra, dtype=jnp.float32)

    def _normalize_designs(self, design: jnp.ndarray) -> np.ndarray:
        arr = np.asarray(design)
        if arr.ndim == 4 and arr.shape[1:] == (1, 12, 12):
            arr = arr[:, 0]
        elif arr.ndim == 3 and arr.shape[-2:] == (12, 12):
            pass
        elif arr.ndim == 2 and arr.shape == (12, 12):
            arr = arr[None, :, :]
        elif arr.ndim == 2 and arr.shape[-1] == 144:
            arr = arr.reshape((-1, 12, 12))
        else:
            raise ValueError(f"ExternalEMScorer only supports 12x12 masks, got shape {arr.shape}")
        arr = (arr > 0.5).astype(np.uint8)
        arr[:, 5:7, 0] = 1
        return arr

    def _cache_key(self, mask: np.ndarray) -> str:
        payload = {
            "solver_mode": self.solver_mode,
            "freq": np.asarray(self.freq, dtype=np.float64).round(3).tolist(),
            "spectrum_dim": self.spectrum_dim,
            "matlab_function": self.matlab_function,
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        digest.update(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / self.solver_mode / f"{key}.npz"

    def _load_cache(self, key: str) -> dict[str, object] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return {
                "spectrum": data["spectrum"].astype(np.float32),
                "valid": bool(data["valid"][0]),
                "error": str(data["error"][0]),
            }

    def _save_cache(self, key: str, spectrum: np.ndarray, valid: bool, error: str) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            spectrum=np.asarray(spectrum, dtype=np.float32),
            valid=np.asarray([valid], dtype=bool),
            error=np.asarray([error], dtype="<U1024"),
        )

    def _run_matlab(self, designs: np.ndarray) -> dict[str, np.ndarray | list[str]]:
        with tempfile.TemporaryDirectory(prefix="pixelant_em_") as tmp:
            tmp_dir = Path(tmp)
            input_mat = tmp_dir / "input.mat"
            output_mat = tmp_dir / "output.mat"
            scipy.io.savemat(
                input_mat,
                {
                    "designs": designs.astype(np.uint8),
                    "freq": np.asarray(self.freq, dtype=np.float64),
                    "solver_mode": self.solver_mode,
                    "options": {},
                },
                do_compression=True,
            )
            self._execute_command(input_mat, output_mat)
            if not output_mat.exists():
                raise RuntimeError(f"MATLAB scorer did not create output file: {output_mat}")
            mat = scipy.io.loadmat(output_mat, squeeze_me=True)
            s11_db = np.asarray(mat["s11_db"], dtype=np.float32)
            if s11_db.ndim == 1:
                s11_db = s11_db[None, :]
            valid = np.asarray(mat["valid"], dtype=bool).reshape(-1)
            errors_raw = np.asarray(mat.get("error_message", np.array([""])), dtype=object).reshape(-1)
            errors = [_matlab_cell_to_str(item) for item in errors_raw]
            if len(errors) < designs.shape[0]:
                errors.extend([""] * (designs.shape[0] - len(errors)))
            return {"s11_db": s11_db, "valid": valid, "error_message": errors}

    def _execute_command(self, input_mat: Path, output_mat: Path) -> None:
        command = self.command or "matlab -batch"
        if "{input_mat}" in command or "{output_mat}" in command:
            rendered = command.format(input_mat=str(input_mat), output_mat=str(output_mat), solver_mode=self.solver_mode)
        else:
            input_arg = _matlab_quote(str(input_mat))
            output_arg = _matlab_quote(str(output_mat))
            expression = f"{self.matlab_function}({input_arg},{output_arg})"
            rendered = f'{command} "{expression}"'
        completed = subprocess.run(
            rendered,
            cwd=self.work_dir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "MATLAB EM scorer failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )


@dataclass
class PNGFScorer:
    """External scorer adapter for the paper center-fed PNGF substrate antenna.

    The scorer intentionally does not vendor or import PNGF source.  It writes a
    small `.npz` batch file and invokes a user-supplied command that is expected
    to call an external PNGF build.  The command must create an output `.npz`
    containing either `targets[N,15]` or `s11_re[N,5]`, `s11_im[N,5]`, and
    `directivity[N,5]`, plus optional `valid[N]` and `error_message[N]`.
    """

    command: str | None = None
    work_dir: Path | None = None
    cache_dir: Path | None = None
    timeout_seconds: float | None = None
    bad_spectrum_value: float = 1.0e6
    spectrum_dim: int = PNGF_TARGET_DIM
    freq_hz: np.ndarray | None = None
    solver_mode: str = "paper-center-fed-substrate"

    def __post_init__(self) -> None:
        self.work_dir = Path(self.work_dir or Path.cwd()).resolve()
        self.cache_dir = Path(self.cache_dir or self.work_dir / "eggroll_runs" / "pngf_cache").resolve()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.timeout_seconds = self.timeout_seconds or 3600.0
        self.freq_hz = np.asarray(self.freq_hz if self.freq_hz is not None else PNGF_FREQ_HZ, dtype=np.float64)
        if self.freq_hz.shape != (5,):
            raise ValueError(f"PNGFScorer expects exactly five frequencies, got shape {self.freq_hz.shape}")
        self.last_valid = np.zeros((0,), dtype=bool)
        self.last_error_message: list[str] = []
        self.last_s11_complex = np.zeros((0, 5), dtype=np.complex64)
        self.last_directivity = np.zeros((0, 5), dtype=np.float32)

    def score(self, design: jnp.ndarray) -> jnp.ndarray:
        masks = project_pngf_center_fed_mask_np(np.asarray(design))
        targets = np.full((masks.shape[0], PNGF_TARGET_DIM), self.bad_spectrum_value, dtype=np.float32)
        valid = np.zeros((masks.shape[0],), dtype=bool)
        errors = [""] * masks.shape[0]
        s11 = np.zeros((masks.shape[0], 5), dtype=np.complex64)
        directivity = np.zeros((masks.shape[0], 5), dtype=np.float32)

        misses: list[tuple[int, str, np.ndarray]] = []
        for idx, mask in enumerate(masks):
            key = self._cache_key(mask)
            cached = self._load_cache(key)
            if cached is None:
                misses.append((idx, key, mask))
                continue
            targets[idx] = cached["target"]
            valid[idx] = bool(cached["valid"])
            errors[idx] = str(cached["error"])
            s11[idx] = cached["s11"]
            directivity[idx] = cached["directivity"]

        if misses:
            if self.command is None:
                raise NotImplementedError(
                    "PNGFScorer is configured as an external scorer. Provide command=... with "
                    "{input_npz} and {output_npz} placeholders, or pre-populate the cache."
                )
            missed_masks = np.stack([item[2] for item in misses], axis=0)
            try:
                result = self._run_pngf_batch(missed_masks)
            except Exception as exc:
                result = {
                    "targets": np.full((len(misses), PNGF_TARGET_DIM), self.bad_spectrum_value, dtype=np.float32),
                    "s11": np.zeros((len(misses), 5), dtype=np.complex64),
                    "directivity": np.zeros((len(misses), 5), dtype=np.float32),
                    "valid": np.zeros((len(misses),), dtype=bool),
                    "error_message": [str(exc)] * len(misses),
                }
            for local_idx, (global_idx, key, _mask) in enumerate(misses):
                row_valid = bool(result["valid"][local_idx])
                row_error = str(result["error_message"][local_idx])
                row_target = np.asarray(result["targets"][local_idx], dtype=np.float32)
                if not row_valid or not np.all(np.isfinite(row_target)):
                    row_target = np.full((PNGF_TARGET_DIM,), self.bad_spectrum_value, dtype=np.float32)
                targets[global_idx] = row_target
                valid[global_idx] = row_valid
                errors[global_idx] = row_error
                s11[global_idx] = np.asarray(result["s11"][local_idx], dtype=np.complex64)
                directivity[global_idx] = np.asarray(result["directivity"][local_idx], dtype=np.float32)
                self._save_cache(key, row_target, s11[global_idx], directivity[global_idx], row_valid, row_error)

        self.last_valid = valid
        self.last_error_message = errors
        self.last_s11_complex = s11
        self.last_directivity = directivity
        return jnp.asarray(targets, dtype=jnp.float32)

    def missing_center_fed_matrices(self, pngf_root: str | Path | None = None) -> list[Path]:
        root = Path(pngf_root).resolve() if pngf_root is not None else self.work_dir
        pngf_opt = root / "pngf-opt" if (root / "pngf-opt").exists() else root
        return [pngf_opt / f"Gmat_sub_{idx:02d}.bin" for idx in range(1, 6) if not (pngf_opt / f"Gmat_sub_{idx:02d}.bin").exists()]

    def _cache_key(self, mask: np.ndarray) -> str:
        payload = {
            "scorer": "pngf",
            "solver_mode": self.solver_mode,
            "freq_hz": np.asarray(self.freq_hz, dtype=np.float64).round(3).tolist(),
            "target_dim": PNGF_TARGET_DIM,
            "geometry": "21x21-paper-center-fed-two-axis",
        }
        digest = hashlib.sha256()
        digest.update(json.dumps(payload, sort_keys=True).encode("utf-8"))
        digest.update(np.ascontiguousarray(mask.astype(np.uint8)).tobytes())
        return digest.hexdigest()

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / "paper_center_fed_substrate" / f"{key}.npz"

    def _load_cache(self, key: str) -> dict[str, object] | None:
        path = self._cache_path(key)
        if not path.exists():
            return None
        with np.load(path, allow_pickle=False) as data:
            return {
                "target": data["target"].astype(np.float32),
                "s11": data["s11"].astype(np.complex64),
                "directivity": data["directivity"].astype(np.float32),
                "valid": bool(data["valid"][0]),
                "error": str(data["error"][0]),
            }

    def _save_cache(
        self,
        key: str,
        target: np.ndarray,
        s11: np.ndarray,
        directivity: np.ndarray,
        valid: bool,
        error: str,
    ) -> None:
        path = self._cache_path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            target=np.asarray(target, dtype=np.float32),
            s11=np.asarray(s11, dtype=np.complex64),
            directivity=np.asarray(directivity, dtype=np.float32),
            valid=np.asarray([valid], dtype=bool),
            error=np.asarray([error], dtype="<U2048"),
        )

    def _run_pngf_batch(self, masks: np.ndarray) -> dict[str, np.ndarray | list[str]]:
        with tempfile.TemporaryDirectory(prefix="pixelant_pngf_") as tmp:
            tmp_dir = Path(tmp)
            input_npz = tmp_dir / "input.npz"
            output_npz = tmp_dir / "output.npz"
            np.savez_compressed(
                input_npz,
                masks=masks.astype(np.uint8),
                freq_hz=np.asarray(self.freq_hz, dtype=np.float64),
                solver_mode=np.asarray([self.solver_mode]),
                geometry=np.asarray(["21x21-paper-center-fed-two-axis"]),
            )
            self._execute_pngf_command(input_npz, output_npz)
            if not output_npz.exists():
                raise RuntimeError(f"PNGF scorer did not create output file: {output_npz}")
            with np.load(output_npz, allow_pickle=False) as data:
                if "targets" in data:
                    targets = np.asarray(data["targets"], dtype=np.float32)
                    if targets.ndim == 1:
                        targets = targets.reshape(1, PNGF_TARGET_DIM)
                    re, im, directivity_jax = _unpack_targets_numpy(targets)
                    s11 = re + 1j * im
                    directivity = directivity_jax
                else:
                    if "s11_complex" in data:
                        s11 = np.asarray(data["s11_complex"], dtype=np.complex64)
                        if s11.ndim == 1:
                            s11 = s11.reshape(1, 5)
                        targets = pack_pngf_targets(s11=s11, directivity=np.asarray(data["directivity"], dtype=np.float32))
                    else:
                        s11_re = np.asarray(data["s11_re"], dtype=np.float32)
                        s11_im = np.asarray(data["s11_im"], dtype=np.float32)
                        directivity = np.asarray(data["directivity"], dtype=np.float32)
                        if s11_re.ndim == 1:
                            s11_re = s11_re.reshape(1, 5)
                            s11_im = s11_im.reshape(1, 5)
                            directivity = directivity.reshape(1, 5)
                        s11 = s11_re.astype(np.float32) + 1j * s11_im.astype(np.float32)
                        targets = pack_pngf_targets(s11_re=s11_re, s11_im=s11_im, directivity=directivity)
                    directivity = np.asarray(data["directivity"], dtype=np.float32)
                    if directivity.ndim == 1:
                        directivity = directivity.reshape(1, 5)
                valid = np.asarray(data["valid"], dtype=bool).reshape(-1) if "valid" in data else np.ones((masks.shape[0],), dtype=bool)
                errors_raw = np.asarray(data["error_message"], dtype=str).reshape(-1) if "error_message" in data else np.asarray([""] * masks.shape[0])
            if targets.shape != (masks.shape[0], PNGF_TARGET_DIM):
                raise ValueError(f"PNGF scorer returned targets shape {targets.shape}, expected {(masks.shape[0], PNGF_TARGET_DIM)}")
            if len(errors_raw) < masks.shape[0]:
                errors = list(errors_raw) + [""] * (masks.shape[0] - len(errors_raw))
            else:
                errors = [str(item) for item in errors_raw[: masks.shape[0]]]
            if valid.shape[0] < masks.shape[0]:
                valid = np.pad(valid, (0, masks.shape[0] - valid.shape[0]), constant_values=False)
            return {
                "targets": targets.astype(np.float32),
                "s11": np.asarray(s11, dtype=np.complex64),
                "directivity": np.asarray(directivity, dtype=np.float32),
                "valid": valid[: masks.shape[0]],
                "error_message": errors,
            }

    def _execute_pngf_command(self, input_npz: Path, output_npz: Path) -> None:
        command = self.command
        if command is None:
            raise NotImplementedError("PNGF command is not configured")
        if "{input_npz}" in command or "{output_npz}" in command:
            rendered = command.format(input_npz=str(input_npz), output_npz=str(output_npz), solver_mode=self.solver_mode)
        else:
            rendered = f'{command} "{input_npz}" "{output_npz}"'
        completed = subprocess.run(
            rendered,
            cwd=self.work_dir,
            shell=True,
            text=True,
            capture_output=True,
            timeout=self.timeout_seconds,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                "PNGF scorer failed with exit code "
                f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )


def build_scorer(config: ScorerConfig) -> Scorer:
    if config.kind == "surrogate":
        if config.checkpoint_path is None:
            raise ValueError("Surrogate scoring requires checkpoint_path")
        return SurrogateScorer.from_checkpoint(config.checkpoint_path)
    if config.kind == "external-em":
        return ExternalEMScorer(
            solver_mode=config.solver_mode,
            command=config.command,
            work_dir=config.work_dir,
            cache_dir=config.cache_dir,
            timeout_seconds=config.timeout_seconds,
            bad_spectrum_value=config.bad_spectrum_value,
        )
    if config.kind == "pngf":
        return PNGFScorer(
            command=config.command,
            work_dir=config.work_dir,
            cache_dir=config.cache_dir,
            timeout_seconds=config.timeout_seconds,
            bad_spectrum_value=config.bad_spectrum_value,
        )
    raise ValueError(f"Unknown scorer kind {config.kind!r}")


def _matlab_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _matlab_cell_to_str(value: object) -> str:
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return ""
        if value.size == 1:
            return str(value.reshape(-1)[0])
        return " ".join(str(item) for item in value.reshape(-1))
    return str(value)


def _unpack_targets_numpy(targets: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    arr = np.asarray(targets, dtype=np.float32).reshape(-1, 5, 3)
    return arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
