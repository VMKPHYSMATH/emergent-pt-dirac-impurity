from __future__ import annotations

import csv
import json
import math
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10 fallback
    import tomli as tomllib


def load_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def fmt_float(value: float) -> str:
    return f"{float(value):.12g}"


def slug_float(value: float) -> str:
    return f"{float(value):.0e}".replace("+", "").replace("-", "m")


def write_run_config(path: Path, *, model: dict[str, Any], nrg: dict[str, Any],
                     output_dir: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "[model]",
        f"U = {fmt_float(model['U'])}",
        f"eps_d = {fmt_float(model['eps_d'])}",
        f"delta0 = {fmt_float(model['delta0'])}",
        f"c_delta = {fmt_float(model['c_delta'])}",
        f"g_gamma = {fmt_float(model['g_gamma'])}",
        f"beta0 = {fmt_float(model['beta0'])}",
        f"delta_coh = {fmt_float(model['delta_coh'])}",
        f"gamma_common = {fmt_float(model['gamma_common'])}",
        f'frame = "{model["frame"]}"',
        f"bandwidth = {fmt_float(model['bandwidth'])}",
        f"bath_exponent = {fmt_float(model['bath_exponent'])}",
        f"gamma_edge = {fmt_float(model['gamma_edge'])}",
        "hybridization_V = nan",
        "bath_split = 0.0",
        "reciprocal_hybridization = true",
        "",
        "[nrg]",
        f"Lambda = {fmt_float(nrg['Lambda'])}",
        f"z_shift = {fmt_float(nrg['z_shift'])}",
        f"iterations = {int(nrg['iterations'])}",
        f"nkeep = {int(nrg['nkeep'])}",
        f"min_keep_per_charge = {int(nrg['min_keep_per_charge'])}",
        f'sort_type = "{nrg["sort_type"]}"',
        f"star_intervals = {int(nrg['star_intervals'])}",
        f"overlap_floor = {fmt_float(nrg['overlap_floor'])}",
        f"degeneracy_tolerance = {fmt_float(nrg['degeneracy_tolerance'])}",
        f"residual_tolerance = {fmt_float(nrg['residual_tolerance'])}",
        f"save_levels = {int(nrg['save_levels'])}",
        "",
        "[lehmann]",
        f"enabled = {str(bool(nrg['lehmann_enabled'])).lower()}",
        "omega_min = -1.0",
        "omega_max = 1.0",
        "omega_points = 2001",
        "eta = 0.01",
        "",
        "[output]",
        f'directory = "{output_dir.as_posix()}"',
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def find_julia(explicit: str | None = None) -> str:
    candidates: list[str] = []
    if explicit:
        candidates.append(explicit)
    env = os.environ.get("JULIA")
    if env:
        candidates.append(env)
    candidates.extend([
        str(Path.home() / ".juliaup" / "bin" / "julia"),
        "julia",
    ])
    for candidate in candidates:
        if "/" in candidate:
            if Path(candidate).exists():
                return candidate
        elif shutil.which(candidate):
            return candidate
    raise FileNotFoundError(
        "Julia was not found. Set JULIA=/path/to/julia or install juliaup."
    )


def run_command(command: list[str], cwd: Path, log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
        code = process.wait()
    if code != 0:
        raise RuntimeError(
            f"Command failed with exit code {code}. See {log_path}"
        )


def read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    parsed: list[dict[str, Any]] = []
    for row in rows:
        converted: dict[str, Any] = {}
        for key, value in row.items():
            if value is None:
                converted[key] = value
                continue
            text = value.strip()
            try:
                converted[key] = float(text)
            except ValueError:
                converted[key] = text
        parsed.append(converted)
    return parsed


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: list[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_run_summary(path: Path) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        # RUN_SUMMARY uses both one key per line and semicolon-separated pairs.
        for segment in line.split(";"):
            if "=" not in segment:
                continue
            key, value = segment.split("=", 1)
            key = key.strip().replace(" ", "_")
            value = value.strip()
            try:
                result[key] = float(value)
            except ValueError:
                result[key] = value
    return result


def consecutive_crossing(iterations: list[int], values: list[float],
                         threshold: float, count: int) -> int | None:
    pairs = sorted(zip(iterations, values))
    for pos in range(0, len(pairs) - count + 1):
        window = pairs[pos:pos + count]
        if all(value < threshold for _, value in window):
            return int(window[0][0])
    return None


def sha256(path: Path) -> str:
    digest = __import__("hashlib").sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
