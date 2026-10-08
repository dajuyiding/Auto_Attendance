#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modelcheck —— 一套可复现的「模型成色体检」探针，用来回答三个问题：

  1. 这个 API 背后的模型好不好？          -> L1 视觉探针 + L2 确定性探针
  2. 它是不是被量化过 / 被换成了小模型？   -> L3 分布指纹 + 协议指纹
  3. 多个供应商之间差多少？                -> 横向对比报告 + gallery.html

三层设计（越往后越贵、越定量）：

  L1 视觉 / agentic 探针
      "Generate an SVG of a pelican riding a bicycle"（鹈鹕测试）
      以及它的变体（动画 HTML、负样本控制组）。
      产出：可渲染的 SVG/HTML + 自动结构打分 + 人工看图。
      为什么有效：鹈鹕骑自行车这个组合在训练集里几乎不存在，
      模型必须真的把「鸟坐在车上、脚踩踏板、手扶车把」的空间关系推出来。

  L2 确定性能力探针
      多位数乘法、字母计数、严格 JSON、精确指令遵循、长列表不截断、
      中文长文、needle-in-haystack、修 bug 并真实执行。
      产出：pass/fail，可复现，无主观性。

  L3 分布与协议指纹（量化检测的核心）
      - 首 token logprob 的 top-1 概率 / 熵：量化会让分布变"糊"。
      - logit 浮点精度反推：仅凭 20 个 logprob 判断 logits 是
        FP32 / BF16 / FP16 / FP8（参考 ICLR Blogposts 2026
        "Extracting Model Precision from 20 Logprobs"）。
      - tokenizer 指纹：固定字符串的 prompt_tokens 必须与标称模型一致。
      - 漂移基线：把本次 logprob 存成 baseline，之后重测做双样本检验
        （参考 arXiv:2512.03816 "Log Probability Tracking of LLM APIs"）。

只依赖 Python 标准库。装了 playwright 就能把 SVG/HTML 渲染成 PNG，
方便肉眼比对，也能做「墨迹覆盖率」这类像素级检查。

用法：
    python3 modelcheck.py --targets targets.example.json --out runs/latest
    python3 modelcheck.py --targets t.json --only visual --jobs 4
    python3 modelcheck.py --selftest          # 不需要 API key，验证打分器
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import datetime as _dt
import hashlib
import json
import math
import os
import random
import re
import statistics
import struct
import subprocess
import sys
import textwrap
import time
import urllib.error
import urllib.request
import xml.etree.ElementTree as ET

__version__ = "0.1.0"


# --------------------------------------------------------------------------
# 基础设施：数据模型 + HTTP
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Target:
    """一个 OpenAI 兼容的端点。"""
    name: str
    base_url: str
    api_key: str = ""
    model: str = ""
    note: str = ""
    extra_body: dict = dataclasses.field(default_factory=dict)
    headers: dict = dataclasses.field(default_factory=dict)

    @property
    def slug(self) -> str:
        # 保留 Unicode 词字符（中文名也能成为可读的目录名），
        # 全部被过滤掉时退化成稳定哈希，避免不同目标撞成同一个 slug 而串数据。
        s = re.sub(r"[^\w.-]+", "-", self.name or "", flags=re.UNICODE).strip("-")
        if not s:
            s = "t-" + hashlib.md5((self.name or "").encode("utf-8")).hexdigest()[:8]
        return s

    def url(self, path: str = "/chat/completions") -> str:
        return self.base_url.rstrip("/") + path


@dataclasses.dataclass
class CallResult:
    """一次 API 调用的结果。ok=False 时 error 里是原因。"""
    ok: bool
    text: str = ""
    reasoning: str = ""
    logprobs: list = dataclasses.field(default_factory=list)  # 每个生成位置一个 dict
    usage: dict = dataclasses.field(default_factory=dict)
    model_echo: str = ""
    finish_reason: str = ""
    latency_ms: int = 0
    error: str = ""
    raw: dict = dataclasses.field(default_factory=dict)


def _post_json(url: str, payload: dict, headers: dict, timeout: int = 180):
    """极简 POST，返回 (status, json_or_text)。status=0 表示网络层失败。"""
    data = json.dumps(payload).encode("utf-8")
    hdrs = {"Content-Type": "application/json", "Accept": "application/json"}
    hdrs.update(headers)
    req = urllib.request.Request(url, data=data, headers=hdrs, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = resp.read().decode("utf-8", "replace")
            try:
                return resp.status, json.loads(body)
            except json.JSONDecodeError:
                return resp.status, body
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", "replace")
        try:
            return e.code, json.loads(body)
        except json.JSONDecodeError:
            return e.code, body
    except Exception as e:  # 网络 / 超时 / TLS
        return 0, f"{type(e).__name__}: {e}"


def call_chat(target: Target, messages: list, *, max_tokens: int = 1024,
              temperature: float = 0.0, want_logprobs: bool = False,
              top_logprobs: int = 20, stop=None, timeout: int = 180,
              seed: int | None = None) -> CallResult:
    """调用 OpenAI 兼容的 /chat/completions，把各家差异抹平成 CallResult。"""
    payload = {
        "model": target.model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }
    if stop:
        payload["stop"] = stop
    if seed is not None:
        payload["seed"] = seed
    if want_logprobs:
        payload["logprobs"] = True
        payload["top_logprobs"] = max(0, min(20, top_logprobs))
    payload.update(target.extra_body or {})

    headers = {"Authorization": f"Bearer {target.api_key}"} if target.api_key else {}
    headers.update(target.headers or {})

    t0 = time.time()
    status, body = _post_json(target.url(), payload, headers, timeout=timeout)
    dt = int((time.time() - t0) * 1000)

    if status == 0:
        return CallResult(ok=False, error=str(body), latency_ms=dt)
    if status != 200 or not isinstance(body, dict):
        msg = body.get("error", body) if isinstance(body, dict) else body
        return CallResult(ok=False, error=f"HTTP {status}: {str(msg)[:400]}", latency_ms=dt)

    try:
        choice = (body.get("choices") or [{}])[0]
        msg = choice.get("message") or {}
        text = msg.get("content") or ""
        if isinstance(text, list):  # 少数网关返回分段 content
            text = "".join(p.get("text", "") for p in text if isinstance(p, dict))
        reasoning = msg.get("reasoning_content") or msg.get("reasoning") or ""
        lp_obj = choice.get("logprobs") or {}
        return CallResult(
            ok=True, text=text, reasoning=reasoning,
            logprobs=lp_obj.get("content") or [],
            usage=body.get("usage") or {}, model_echo=body.get("model", ""),
            finish_reason=choice.get("finish_reason", ""), latency_ms=dt, raw=body,
        )
    except Exception as e:
        return CallResult(ok=False, error=f"解析响应失败: {e}", latency_ms=dt)


# --------------------------------------------------------------------------
# L1：SVG / HTML 结构分析
# --------------------------------------------------------------------------

_TAG_RE = re.compile(r"<\s*(/?)\s*([a-zA-Z][\w:.-]*)((?:[^<>\"']|\"[^\"]*\"|'[^']*')*?)(/?)\s*>", re.S)
_ATTR_RE = re.compile(r"([\w:.-]+)\s*=\s*(?:\"([^\"]*)\"|'([^']*)')")
_NUM_RE = re.compile(r"-?\d*\.?\d+(?:[eE][-+]?\d+)?")
_TRANSFORM_RE = re.compile(r"(translate|scale|matrix|rotate)\s*\(([^)]*)\)")
_ROUND_TAGS = {"circle", "ellipse"}
_SHAPE_TAGS = {"circle", "ellipse", "rect", "line", "polygon", "polyline", "path"}
_WARM_COLORS = {"orange", "gold", "yellow", "darkorange", "coral", "khaki", "amber",
                "ffa500", "ffd700", "ffff00", "ffc107", "ffb347", "ffe066", "f4a460"}


def _attrs(raw: str) -> dict:
    out = {}
    for m in _ATTR_RE.finditer(raw or ""):
        out[m.group(1).lower()] = m.group(2) if m.group(2) is not None else m.group(3)
    return out


def _floats(s: str) -> list:
    return [float(x) for x in _NUM_RE.findall(s or "")]


def _parse_transform(t: str):
    """把 transform 折成一个 (scale, tx, ty) 近似（只处理常见的 translate/scale/matrix）。"""
    s, tx, ty = 1.0, 0.0, 0.0
    for kind, args in _TRANSFORM_RE.findall(t or ""):
        v = _floats(args)
        if kind == "translate" and v:
            tx += s * v[0]
            ty += s * (v[1] if len(v) > 1 else 0.0)
        elif kind == "scale" and v:
            ty *= v[1] if len(v) > 1 else v[0]
            tx *= v[0]
            s *= v[0]
        elif kind == "matrix" and len(v) >= 6:
            s *= abs(v[0]) or 1.0
            tx += v[4] * s
            ty += v[5] * s
    return s, tx, ty


@dataclasses.dataclass
class Shape:
    tag: str
    attrs: dict
    bbox: tuple | None          # 画布坐标系下的 (x0, y0, x1, y1)
    fill: str
    stroke: str
    has_curve: bool = False     # path 里出现了 Q/C/A/S 曲线指令
    closed: bool = False

    @property
    def cx(self):
        return (self.bbox[0] + self.bbox[2]) / 2 if self.bbox else None

    @property
    def cy(self):
        return (self.bbox[1] + self.bbox[3]) / 2 if self.bbox else None

    @property
    def w(self):
        return self.bbox[2] - self.bbox[0] if self.bbox else 0.0

    @property
    def h(self):
        return self.bbox[3] - self.bbox[1] if self.bbox else 0.0


@dataclasses.dataclass
class SvgAnalysis:
    ok: bool = False
    xml_valid: bool = False
    reason: str = ""
    n_elements: int = 0
    n_shapes: int = 0
    n_paths: int = 0
    n_groups: int = 0
    n_text: int = 0
    n_comments: int = 0
    n_curves: int = 0
    has_viewbox: bool = False
    has_width_height: bool = False
    canvas_w: float = 0.0
    canvas_h: float = 0.0
    distinct_colors: int = 0
    shapes: list = dataclasses.field(default_factory=list)
    wheel_pair: tuple | None = None
    body: "Shape | None" = None
    head: "Shape | None" = None
    beak: "Shape | None" = None
    n_legs: int = 0
    n_frame_lines: int = 0
    bg_coverage: float = 0.0
    duplicate_ratio: float = 0.0
    chars: int = 0
    raw: str = ""

def analyze_svg(text: str) -> SvgAnalysis:
    """把一段 SVG 源码变成可打分的结构特征。容忍轻微畸形（小模型常写不闭合标签）。"""
    a = SvgAnalysis(raw=text or "")
    a.chars = len(text or "")
    if not text or "<svg" not in text.lower():
        a.reason = "没有找到 <svg> 元素"
        return a

    a.n_comments = len(re.findall(r"<!--[\s\S]*?-->", text))
    try:
        ET.fromstring(text)
        a.xml_valid = True
    except ET.ParseError:
        a.xml_valid = False
    a.ok = True

    # 用正则扫描所有元素（对不闭合标签也鲁棒），同时维护 transform 栈
    stack = [(1.0, 0.0, 0.0)]
    root_attrs: dict = {}
    first_svg_seen = False
    for m in _TAG_RE.finditer(text):
        closing, tag, raw_attrs, selfclose = m.group(1), m.group(2).lower(), m.group(3), m.group(4)
        tag = tag.split(":")[-1]
        if closing:
            if tag == "g" and len(stack) > 1:
                stack.pop()
            continue
        a.n_elements += 1
        at = _attrs(raw_attrs)
        if tag == "svg" and not first_svg_seen:
            first_svg_seen = True
            root_attrs = at
        if tag in ("style", "script"):
            continue
        if tag == "text":
            a.n_text += 1
        if tag == "path" and re.search(r"[QqCcAaSs]", (at.get("d") or "")):
            a.n_curves += 1

        parent_s, parent_tx, parent_ty = stack[-1]
        s, tx, ty = _parse_transform(at.get("transform", ""))
        net = (parent_s * s, parent_tx + parent_s * tx, parent_ty + parent_s * ty)
        if tag == "g" and not selfclose:
            stack.append(net)
            a.n_groups += 1

        if tag in _SHAPE_TAGS:
            bb = _shape_bbox(tag, at)
            if bb:
                s_, tx_, ty_ = net
                bb = (bb[0] * s_ + tx_, bb[1] * s_ + ty_, bb[2] * s_ + tx_, bb[3] * s_ + ty_)
            a.shapes.append(Shape(
                tag=tag, attrs=at, bbox=bb,
                fill=(at.get("fill") or "").lower(), stroke=(at.get("stroke") or "").lower(),
                has_curve=bool(re.search(r"[QqCcAaSs]", at.get("d") or "")),
                closed=tag in ("circle", "ellipse", "rect", "polygon") or
                       bool(re.search(r"[Zz]", at.get("d") or "")),
            ))
            a.n_shapes += 1

    # 画布
    vb = _floats(root_attrs.get("viewbox", ""))
    if len(vb) == 4:
        a.has_viewbox = True
        a.canvas_w, a.canvas_h = vb[2], vb[3]
    w, h = root_attrs.get("width"), root_attrs.get("height")
    if w and h:
        a.has_width_height = True
        wf, hf = _floats(w), _floats(h)
        if wf and hf and not a.canvas_w:
            a.canvas_w, a.canvas_h = wf[0], hf[0]
    if not a.canvas_w:
        a.canvas_w = a.canvas_h = 100.0
    a.n_paths = sum(1 for s in a.shapes if s.tag == "path")

    # 颜色
    colors = set()
    for s in a.shapes:
        for c in (s.fill, s.stroke):
            if c and c not in ("none", "transparent"):
                colors.add(c)
    a.distinct_colors = len(colors)

    # 背景覆盖：一个几乎铺满画布的 rect
    W, H = a.canvas_w, a.canvas_h
    for s in a.shapes:
        if s.tag == "rect" and s.bbox and W and H:
            cov = (s.w * s.h) / (W * H)
            if s.bbox[0] <= 0.02 * W and s.bbox[1] <= 0.02 * H:
                a.bg_coverage = max(a.bg_coverage, min(1.0, cov))

    # 重复元素比例（循环/复制粘贴式输出的指纹）
    if a.shapes:
        keys = [(s.tag, tuple(sorted((k, v) for k, v in s.attrs.items() if k != "transform")))
                for s in a.shapes]
        a.duplicate_ratio = 1.0 - len(set(keys)) / len(keys)

    _find_pelican_and_bike(a)
    return a


def _shape_bbox(tag: str, at: dict):
    """返回元素在自身坐标系下的包围盒。"""
    try:
        if tag == "circle":
            cx, cy, r = float(at.get("cx", 0)), float(at.get("cy", 0)), float(at.get("r", 0))
            return (cx - r, cy - r, cx + r, cy + r)
        if tag == "ellipse":
            cx, cy = float(at.get("cx", 0)), float(at.get("cy", 0))
            rx, ry = float(at.get("rx", 0)), float(at.get("ry", 0))
            return (cx - rx, cy - ry, cx + rx, cy + ry)
        if tag == "rect":
            x, y = float(at.get("x", 0)), float(at.get("y", 0))
            w, h = float(at.get("width", 0)), float(at.get("height", 0))
            return (x, y, x + w, y + h)
        if tag == "line":
            x1, y1 = float(at.get("x1", 0)), float(at.get("y1", 0))
            x2, y2 = float(at.get("x2", 0)), float(at.get("y2", 0))
            return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))
        if tag in ("polygon", "polyline"):
            v = _floats(at.get("points", ""))
            xs, ys = v[0::2], v[1::2]
            if not xs or not ys:
                return None
            return (min(xs), min(ys), max(xs), max(ys))
        if tag == "path":
            v = _floats(at.get("d", ""))
            xs, ys = v[0::2], v[1::2]
            if len(xs) < 2 or len(ys) < 2:
                return None
            return (min(xs), min(ys), max(xs), max(ys))
    except (TypeError, ValueError):
        return None
    return None


def _find_pelican_and_bike(a: SvgAnalysis):
    """在结构特征上找「两个轮子 / 车架 / 鹈鹕身体 / 头 / 喙 / 腿」。"""
    W, H = a.canvas_w, a.canvas_h
    if not W or not H:
        return
    shapes = [s for s in a.shapes if s.bbox]

    # --- 轮子：圆或接近圆的椭圆，够大，位置偏下，且左右成对 ---
    cands = []
    for s in shapes:
        if s.tag not in _ROUND_TAGS:
            continue
        rx, ry = s.w / 2, s.h / 2
        if rx <= 0 or ry <= 0:
            continue
        roundness = min(rx, ry) / max(rx, ry)          # 1.0 = 正圆
        r = (rx + ry) / 2
        if roundness < 0.55 or r < 0.04 * W or r > 0.42 * W:
            continue
        if s.cy is None or s.cy < 0.45 * H:            # 轮子应该在下半部分
            continue
        cands.append((s, r, roundness))
    best = None
    for i in range(len(cands)):
        for j in range(i + 1, len(cands)):
            s1, r1, q1 = cands[i]
            s2, r2, q2 = cands[j]
            rmean = (r1 + r2) / 2
            ratio = min(r1, r2) / max(r1, r2)
            dx = abs(s1.cx - s2.cx)
            dy = abs(s1.cy - s2.cy)
            if ratio < 0.62 or dx < 1.15 * rmean or dy > 0.6 * rmean:
                continue
            quality = ratio * ((q1 + q2) / 2) * min(1.0, dx / (2.6 * rmean))
            if best is None or quality > best[0]:
                best = (quality, s1, s2, rmean, dx)
    if best:
        a.wheel_pair = (best[1], best[2], best[3], best[4])

    # --- 车架：连接两轮之间的线 / 路径 ---
    if a.wheel_pair:
        s1, s2, rmean, dx = a.wheel_pair
        x_lo, x_hi = min(s1.cx, s2.cx), max(s1.cx, s2.cx)
        y_wheel = (s1.cy + s2.cy) / 2
        for s in shapes:
            if s.tag not in ("line", "path", "polyline", "rect"):
                continue
            if s.cx is None:
                continue
            inside = x_lo - 0.2 * dx <= s.cx <= x_hi + 0.2 * dx
            above = s.cy < y_wheel + 0.15 * rmean
            near = abs(s.cy - y_wheel) < 1.6 * rmean
            if inside and above and near:
                a.n_frame_lines += 1

    # --- 鹈鹕身体：轮子上方、水平落在两轮之间的一个「块状」形状 ---
    if a.wheel_pair:
        s1, s2, rmean, dx = a.wheel_pair
        x_lo, x_hi = min(s1.cx, s2.cx), max(s1.cx, s2.cx)
        y_wheel = (s1.cy + s2.cy) / 2
        bodies = []
        for s in shapes:
            if s.tag not in ("ellipse", "circle", "path", "polygon"):
                continue
            if s.cx is None or s.w <= 0.05 * W:
                continue
            if x_lo - 0.4 * dx <= s.cx <= x_hi + 0.4 * dx and s.cy < y_wheel - 0.15 * rmean:
                if s.w * s.h > 0.004 * W * H:
                    bodies.append(s)
        if bodies:
            a.body = max(bodies, key=lambda s: s.w * s.h)

    # --- 头 / 喙 / 腿 ---
    if a.body and a.body.bbox:
        bx0, by0, bx1, by1 = a.body.bbox
        head_cands = []
        for s in shapes:
            if s is a.body or s.tag not in ("circle", "ellipse", "path", "polygon"):
                continue
            if s.cx is None:
                continue
            if s.cy < by0 + 0.15 * (by1 - by0) and abs(s.cx - a.body.cx) < 1.6 * a.body.w:
                # 头必须是个「有份量」的块，不能是眼睛那种小圆点
                if 0.10 * a.body.w < s.w < 1.3 * a.body.w:
                    head_cands.append(s)
        if head_cands:
            a.head = min(head_cands, key=lambda s: s.cy)

        if a.head and a.head.bbox:
            hy = a.head.cy
            hw = a.head.w or 1.0
            beak_cands = []
            for s in shapes:
                if s is a.head or s is a.body or s.cx is None:
                    continue
                aspect = s.w / s.h if s.h > 0 else 0.0
                warm = any(c in _WARM_COLORS for c in (s.fill, s.stroke))
                rightish = s.cx > a.head.cx - 0.2 * hw
                near = abs(s.cy - hy) < 2.2 * max(a.head.h, 1e-6)
                if rightish and near and (aspect >= 1.5 or warm):
                    beak_cands.append(s)
            if beak_cands:
                a.beak = max(beak_cands, key=lambda s: s.w * (
                    2.0 if any(c in _WARM_COLORS for c in (s.fill, s.stroke)) else 1.0))

        # 腿：从身体下缘往踏板/车架方向的细长元素
        if a.wheel_pair:
            y_wheel = (a.wheel_pair[0].cy + a.wheel_pair[1].cy) / 2
            for s in shapes:
                if s.tag not in ("line", "path", "polyline"):
                    continue
                if s.cx is None or s.cy is None:
                    continue
                if by1 - 0.35 * (by1 - by0) <= s.cy < y_wheel and abs(s.cx - a.body.cx) < 0.9 * a.body.w:
                    a.n_legs += 1


# --------------------------------------------------------------------------
# L1：评分规则（全部可解释，报告里会打印每条扣分理由）
# --------------------------------------------------------------------------

def _mk(crit, key, label, score, weight, note=""):
    crit.append({"key": key, "label": label,
                 "score": round(max(0.0, min(1.0, score)), 3),
                 "weight": weight, "note": note})


def _finish(crit) -> dict:
    wsum = sum(c["weight"] for c in crit) or 1.0
    total = sum(c["score"] * c["weight"] for c in crit) / wsum
    return {"score": round(total, 4), "criteria": crit}


def score_svg(a: SvgAnalysis) -> dict:
    """给一张 SVG 打 0..1 的分。返回 {score, criteria:[{key,label,score,weight,note}]}"""
    crit = []

    # 1) 是不是一份能用的 SVG
    if not a.ok:
        _mk(crit, "valid_svg", "SVG 合法性", 0.0, 1.0, a.reason)
        return _finish(crit)
    if not (a.has_viewbox or a.has_width_height):
        _mk(crit, "valid_svg", "SVG 合法性", 0.6, 0.10,
            "有 <svg> 但既无 viewBox 也无 width/height，尺寸未定义")
    elif not a.xml_valid:
        _mk(crit, "valid_svg", "SVG 合法性", 0.7, 0.10, "有尺寸声明，但 XML 不合法（标签未闭合等）")
    else:
        _mk(crit, "valid_svg", "SVG 合法性", 1.0, 0.10, "有 viewBox/尺寸声明且 XML 合法")

    # 2) 画布自洽
    if a.has_viewbox and a.has_width_height:
        _mk(crit, "canvas", "画布自洽", 1.0, 0.05, "viewBox 与 width/height 同时存在")
    elif a.has_viewbox:
        _mk(crit, "canvas", "画布自洽", 1.0, 0.05, "有 viewBox（画布自洽）")
    elif a.has_width_height:
        _mk(crit, "canvas", "画布自洽", 0.6, 0.05, "只有 width/height，没有 viewBox（缩放会出问题）")
    else:
        _mk(crit, "canvas", "画布自洽", 0.3, 0.05, "画布未声明")

    # 3) 车轮
    if not a.wheel_pair:
        _mk(crit, "wheels", "两个车轮", 0.0, 0.18, "没有找到两个尺寸相近、够大、位于下方的圆形")
    else:
        s1, s2, rmean, dx = a.wheel_pair
        r1 = (s1.w / 2 + s1.h / 2) / 2
        r2 = (s2.w / 2 + s2.h / 2) / 2
        ratio = min(r1, r2) / max(r1, r2)
        q = statistics.mean([min(s1.w, s1.h) / max(s1.w, s1.h, 1e-6),
                             min(s2.w, s2.h) / max(s2.w, s2.h, 1e-6)])
        s = 0.45 + 0.25 * ratio + 0.15 * min(1.0, dx / (2.4 * rmean)) + 0.15 * q
        _mk(crit, "wheels", "两个车轮", s, 0.18,
            f"半径 {r1:.0f}/{r2:.0f}（比 {ratio:.2f}），间距 {dx:.0f}，圆度 {q:.2f}")

    # 4) 车架
    nf = a.n_frame_lines
    s = 0.0 if nf == 0 else (0.4 if nf == 1 else 0.65 if nf == 2 else 0.8 if nf == 3 else 1.0)
    _mk(crit, "frame", "车架结构", s, 0.12,
        "两轮之间找不到连接结构" if nf == 0 else f"两轮之间有 {nf} 个连接件（车架/前叉/座管）")

    # 5) 鹈鹕身体
    if not a.body:
        _mk(crit, "body", "鹈鹕身体", 0.0, 0.12, "轮子上方、两轮之间没有块状身体")
    else:
        area = a.body.w * a.body.h / max(a.canvas_w * a.canvas_h, 1e-6)
        aspect = a.body.w / a.body.h if a.body.h else 0
        s = 0.5 + (0.25 if 0.008 <= area <= 0.30 else 0.0) + (0.25 if 0.8 <= aspect <= 3.5 else 0.0)
        _mk(crit, "body", "鹈鹕身体", s, 0.12, f"身体占画布 {area*100:.1f}%，宽高比 {aspect:.2f}")

    # 6) 喙
    if not a.beak:
        _mk(crit, "beak", "喙 / 喉囊", 0.0, 0.10, "头部附近没有找到长条状或暖色的喙")
    else:
        warm = any(c in _WARM_COLORS for c in (a.beak.fill, a.beak.stroke))
        aspect = a.beak.w / a.beak.h if a.beak.h else 0
        s = 0.5 + (0.3 if warm else 0.0) + (0.2 if aspect >= 1.5 else 0.0)
        _mk(crit, "beak", "喙 / 喉囊", s, 0.10,
            f"喙宽高比 {aspect:.2f}，{'暖色（鹈鹕喙的典型画法）' if warm else '非暖色'}")

    # 7) 腿 / 与车的接触
    s = min(1.0, a.n_legs / 2.0)
    _mk(crit, "legs", "腿 / 踩在车上", s, 0.10,
        f"身体到踏板方向有 {a.n_legs} 条腿线" if a.n_legs else "没有腿：鸟浮在车上，空间关系没建立")

    # 8) 头颈
    if not a.head:
        _mk(crit, "head", "头 / 颈", 0.0, 0.08, "身体上方没有找到头部")
    else:
        s = 0.6
        if a.body and a.head.w < a.body.w:
            s += 0.2
        if a.body and a.head.cy is not None and a.body.cy is not None and a.head.cy < a.body.cy:
            s += 0.2
        _mk(crit, "head", "头 / 颈", s, 0.08,
            f"头宽 {a.head.w:.0f} vs 身体宽 {a.body.w:.0f}" if a.body else "找到头部")

    # 9) 技法
    s, bits = 0.0, []
    if a.n_curves:
        s += 0.3
        bits.append(f"{a.n_curves} 条曲线路径")
    if a.n_groups:
        s += 0.15
        bits.append(f"{a.n_groups} 个 <g> 分组")
    if re.search(r"stroke-width\s*=", a.raw):
        s += 0.15
        bits.append("用了 stroke-width")
    if a.has_viewbox:
        s += 0.2
        bits.append("用了 viewBox")
    if a.n_elements >= 15:
        s += 0.2
        bits.append(f"{a.n_elements} 个元素")
    _mk(crit, "technique", "技法丰富度", min(1.0, s), 0.08, "、".join(bits) or "只有最基本图元")

    # 10) 干净度（扣分项）
    s, notes = 1.0, []
    if a.n_text > 0:
        s -= 0.4
        notes.append(f"用 {a.n_text} 个 <text> 标注代替画画")
    if a.distinct_colors <= 1:
        s -= 0.3
        notes.append("只有一种颜色")
    if a.duplicate_ratio > 0.5:
        s -= 0.4
        notes.append(f"{a.duplicate_ratio*100:.0f}% 的元素完全重复（循环式输出）")
    if a.bg_coverage > 0.9 and a.n_shapes < 10:
        s -= 0.3
        notes.append("一个背景矩形铺满画布，主体内容很少")
    if a.chars < 400:
        s -= 0.3
        notes.append(f"总共只有 {a.chars} 字符，投入极低")
    if not a.xml_valid:
        s -= 0.25
        notes.append("XML 不合法")
    _mk(crit, "cleanliness", "干净度", s, 0.07, "；".join(notes) or "没有明显退化特征")

    return _finish(crit)


def score_svg_generic(a: SvgAnalysis) -> dict:
    """通用 SVG 质量分：不假设画的是什么，只看「像不像一张认真画出来的图」。

    用于控制组（房子）和非自行车类探针（独轮车）。
    """
    crit = []
    if not a.ok:
        _mk(crit, "valid_svg", "SVG 合法性", 0.0, 0.20, a.reason)
        return _finish(crit)
    _mk(crit, "valid_svg", "SVG 合法性",
        1.0 if (a.has_viewbox or a.has_width_height) and a.xml_valid else
        (0.7 if a.has_viewbox or a.has_width_height else 0.4), 0.20,
        ("有 viewBox/尺寸且 XML 合法" if a.xml_valid else "XML 不合法"))
    _mk(crit, "canvas", "画布自洽", 1.0 if a.has_viewbox else 0.5, 0.10,
        "有 viewBox" if a.has_viewbox else "没有 viewBox")
    n = a.n_elements
    s = 0.0 if n < 4 else (0.3 if n < 8 else 0.6 if n < 12 else 0.85 if n < 18 else 1.0)
    _mk(crit, "complexity", "结构复杂度", s, 0.20, f"{n} 个元素、{a.n_shapes} 个图元")
    c = a.distinct_colors
    _mk(crit, "color", "用色", 0.0 if c <= 1 else (0.5 if c == 2 else 1.0), 0.15,
        f"{c} 种颜色")
    tech, bits = 0.0, []
    if a.n_curves:
        tech += 0.3; bits.append(f"{a.n_curves} 条曲线")
    if a.n_groups:
        tech += 0.2; bits.append(f"{a.n_groups} 个 <g>")
    if re.search(r"stroke-width\s*=", a.raw):
        tech += 0.2; bits.append("stroke-width")
    if a.has_viewbox:
        tech += 0.3; bits.append("viewBox")
    _mk(crit, "technique", "技法丰富度", min(1.0, tech), 0.20, "、".join(bits) or "只有最基本图元")
    s, notes = 1.0, []
    if a.n_text > 2:
        s -= 0.4; notes.append(f"{a.n_text} 个 <text> 当主体")
    if a.duplicate_ratio > 0.6:
        s -= 0.4; notes.append("大量元素完全重复")
    if a.chars < 300:
        s -= 0.3; notes.append("投入极低")
    if not a.xml_valid:
        s -= 0.25; notes.append("XML 不合法")
    _mk(crit, "cleanliness", "干净度", s, 0.15, "；".join(notes) or "没有退化特征")
    return _finish(crit)


def score_html(text: str) -> dict:
    """给「自包含 HTML 动画」打分。"""
    crit = []
    low = (text or "").lower()

    is_html = ("<html" in low) or ("<!doctype" in low) or ("<body" in low)
    _mk(crit, "html", "是完整 HTML", 1.0 if is_html else 0.35, 0.10,
        "完整 HTML 文档" if is_html else "不是完整 HTML 文档")

    ext = re.findall(r'(?:src|href)\s*=\s*["\'](?:https?:)?//', low)
    _mk(crit, "selfcontained", "自包含", 1.0 - min(1.0, len(ext) / 3.0), 0.10,
        "无外部依赖" if not ext else f"引用了 {len(ext)} 个外部资源（无法离线渲染）")

    m = re.search(r"<svg[\s\S]*?</svg>", text or "", re.I)
    if m:
        sub = score_svg(analyze_svg(m.group(0)))
        _mk(crit, "svg_inside", "内嵌 SVG 质量", sub["score"], 0.40,
            f"内嵌 SVG 结构分 {sub['score']*100:.0f}")
    else:
        _mk(crit, "svg_inside", "内嵌 SVG 质量", 0.0, 0.40, "HTML 里没有 <svg>")

    anim, bits = 0.0, []
    if "@keyframes" in low:
        anim += 0.4
        bits.append("CSS @keyframes")
    if re.search(r"<animate(transform|motion)?\b", low):
        anim += 0.3
        bits.append("SMIL <animate>")
    if "requestanimationframe" in low:
        anim += 0.2
        bits.append("requestAnimationFrame")
    if re.search(r"transition\s*:", low):
        anim += 0.1
        bits.append("CSS transition")
    _mk(crit, "animation", "有动画", min(1.0, anim), 0.20, "、".join(bits) or "没有任何动画机制")

    spin = 0.0
    if re.search(r"rotate\s*\(", low):
        spin += 0.5
    if re.search(r"transform-origin", low) or re.search(r"transform-box", low):
        spin += 0.25
    if re.search(r"wheel|轮", low):
        spin += 0.25
    _mk(crit, "wheel_spin", "车轮转动", min(1.0, spin), 0.20,
        "有旋转动画" if spin >= 0.5 else "没有让车轮转起来的逻辑")

    return _finish(crit)


# --------------------------------------------------------------------------
# L3：logprob 分析 —— 量化检测的核心
# --------------------------------------------------------------------------

def _f32(x: float) -> float:
    """模拟「API 返回的是 32 位浮点」。"""
    try:
        return struct.unpack("<f", struct.pack("<f", float(x)))[0]
    except (OverflowError, struct.error):
        return float(x)


def _build_fp8_set(exp_bits: int, man_bits: int, bias: int):
    """枚举一个 FP8 格式所有可表示的值，用于精确判定「是否可表示」。"""
    vals = set()
    max_e = (1 << exp_bits) - 1
    for e in range(0, max_e):            # 全 1 指数留给 inf/nan
        for m in range(0, 1 << man_bits):
            if e == 0:
                v = (m / (1 << man_bits)) * (2.0 ** (1 - bias))
            else:
                v = (1.0 + m / (1 << man_bits)) * (2.0 ** (e - bias))
            if v == 0:
                continue
            vals.add(v)
            vals.add(-v)
    return vals


_FP8_E4M3 = _build_fp8_set(4, 3, 7)     # 3 个尾数位
_FP8_E5M2 = _build_fp8_set(5, 2, 15)    # 2 个尾数位
_FORMATS = ["fp8_e5m2", "fp8_e4m3", "bf16", "fp16", "fp32"]
_MANTISSA_BITS = {"fp8_e5m2": 2, "fp8_e4m3": 3, "bf16": 7, "fp16": 10, "fp32": 23}


def _representable(x: float, fmt: str) -> bool:
    """x 能否被 fmt 精确表示？"""
    if x == 0.0:
        return True
    if fmt == "fp32":
        return _f32(x) == x
    if fmt == "fp16":
        try:
            return struct.unpack("<e", struct.pack("<e", x))[0] == x
        except (OverflowError, struct.error):
            return False
    if fmt == "bf16":
        bits = struct.unpack("<I", struct.pack("<f", _f32(x)))[0]
        return (bits & 0xFFFF) == 0        # 低 16 位尾数必须为 0
    if fmt == "fp8_e4m3":
        return x in _FP8_E4M3
    if fmt == "fp8_e5m2":
        return x in _FP8_E5M2
    return False


def position_logprobs(logprobs: list, pos_index: int = 0) -> list:
    """取出某个位置上同一批 top-k 的 logprob 值。

    必须是同一次 log-softmax 产生的（即同一个 token 位置的 top-k），
    才能做浮点精度反推。
    """
    if not logprobs or pos_index >= len(logprobs):
        return []
    pos = logprobs[pos_index]
    if not isinstance(pos, dict):
        return []
    vals = []
    for a in (pos.get("top_logprobs") or []):
        v = a.get("logprob") if isinstance(a, dict) else None
        if isinstance(v, (int, float)):
            vals.append(_f32(v))
    if not vals and isinstance(pos.get("logprob"), (int, float)):
        vals = [_f32(pos["logprob"])]
    return vals


def pick_richest_position(logprobs: list) -> int:
    """挑一个 top-k 数量最多、分布最不极端的位置（最适合做精度反推）。"""
    best_i, best_key = 0, (-1, -1.0)
    for i, _ in enumerate(logprobs or []):
        vals = position_logprobs(logprobs, i)
        if len(vals) < 2:
            continue
        key = (len(vals), max(vals) - min(vals))
        if key > best_key:
            best_i, best_key = i, key
    return best_i


_FINE_FORMATS = ["fp8_e5m2", "fp8_e4m3", "bf16", "fp16"]


def _nearest(x: float, fmt: str):
    """返回 fmt 中离 x 最近的可表示值。"""
    if x == 0.0:
        return 0.0
    if fmt == "fp32":
        return _f32(x)
    if fmt == "fp16":
        try:
            return struct.unpack("<e", struct.pack("<e", x))[0]
        except (OverflowError, struct.error):
            return math.copysign(65504.0, x)
    if fmt == "bf16":
        bits = struct.unpack("<I", struct.pack("<f", _f32(x)))[0]
        bits = (bits + 0x8000) & 0xFFFF0000      # 就近舍入到 bf16
        return struct.unpack("<f", struct.pack("<I", bits))[0]
    s = _FP8_E4M3 if fmt == "fp8_e4m3" else _FP8_E5M2
    return min(s, key=lambda v: abs(v - abs(x))) * (1 if x >= 0 else -1)


def _near(x: float, fmt: str, eps: float) -> bool:
    """x 是否落在 fmt 的格点上（允许 eps 的绝对容差）。

    真实 API 返回的 logprob 是 fp32，往返一次必然有 ~1 ulp 的舍入误差，
    所以不能要求「完全相等」，只能要求「离最近格点足够近」。
    """
    return abs(x - _nearest(x, fmt)) <= eps


def _positive_values(fmt: str, lo: float, hi: float) -> list:
    """枚举 fmt 在 [lo, hi] 内的正的可表示值。"""
    if hi <= 0:
        return []
    lo = max(lo, 1e-9)
    if fmt.startswith("fp8"):
        s = _FP8_E4M3 if fmt == "fp8_e4m3" else _FP8_E5M2
        return sorted(v for v in s if lo <= v <= hi)
    mb, clo, chi = {"fp16": (10, -14, 15), "bf16": (7, -126, 127)}[fmt]
    e0 = max(clo, int(math.floor(math.log2(lo))))
    e1 = min(chi, int(math.ceil(math.log2(hi))))
    n = 1 << mb
    out = []
    for e in range(e0, e1 + 1):
        base = 2.0 ** e
        for m in range(n):
            v = (1.0 + m / n) * base
            if lo <= v <= hi:
                out.append(v)
    return out


def _values_in_range(fmt: str, lo: float, hi: float) -> list:
    """fmt 在 [lo, hi] 内所有（含负）可表示值，作为候选「原始 logits」的取值域。"""
    if lo > hi:
        return []
    pos = _positive_values(fmt, max(lo, 0.0), hi)
    neg = [-v for v in _positive_values(fmt, max(-hi, 0.0), max(-lo, 0.0))]
    out = set(pos) | set(neg)
    if lo <= 0.0 <= hi:
        out.add(0.0)
    return sorted(out)


def _find_shift(vals: list, fmt: str, logit_lo: float = -70.0,
                logit_hi: float = 140.0, rel_eps: float = 4e-6):
    """找一个平移量 C，使所有 (val + C) 都落在 fmt 的格点上。

    做法是枚举「第一个 logprob 对应的原始 logit」在 fmt 上的所有可能取值 s，
    令 C = s - vals[0]，再校验其余 logprob 是否同时落在格点上。
    第一个 logprob 是 top-1，对应的 logit 就是最大 logit，通常为正且不大，
    所以枚举区间取 [-70, 140] 足够覆盖真实模型的取值范围。

    返回 (C, 平均残差)；找不到返回 (None, None)。残差是「离最近格点的距离」，
    可以当作置信度看：残差越小、命中越干净。
    """
    if not vals:
        return None, None
    base = vals[0]
    for s in _values_in_range(fmt, logit_lo, logit_hi):
        c = s - base
        if not (-100.0 <= c <= 200.0):
            continue
        res = []
        for v in vals:
            x = _f32(v + c)
            res.append(abs(x - _nearest(x, fmt)))
        if max(res) <= rel_eps * max(1.0, abs(base + c)):
            return c, sum(res) / len(res)
    return None, None


def infer_logit_precision(vals: list, n_use: int = 20) -> dict:
    """仅凭一批 logprob 反推 logits 的浮点精度。

    原理（ICLR Blogposts 2026, "Extracting Model Precision from 20 Logprobs"）：
    logprobs = logits - logsumexp(logits)，即所有 logits 被平移了一个常数 C。
    返回的 logprob 本身是 fp32，但如果原始 logits 是低精度存储的，
    那么必然存在一个 C，使 (logprob_i + C) 在低精度格式里**精确可表示**。
    于是：搜索 C，看哪种格式能同时满足全部 logprob，取尾数位最少的那个。

    注意：只能判断「logits 的精度」，不能判断权重量化方案（int4 等非标准格式
    会回落成 fp32 的结论）。仅对暴露 top-k logprobs 的 API 有效。
    """
    vals = [_f32(v) for v in (vals or []) if isinstance(v, (int, float))]
    if len(vals) < 5:
        return {"ok": False, "reason": f"可用的 logprob 太少（{len(vals)} 个，至少 5 个）"}
    vals = vals[:n_use]

    hits = {}
    for fmt in _FINE_FORMATS:
        c, resid = _find_shift(vals, fmt)
        if c is not None:
            hits[fmt] = {"shift": round(c, 8), "residual": resid}

    if not hits:
        return {"ok": True, "precision": "fp32", "mantissa_bits": 23,
                "all_matching": ["fp32"], "shifts": {}, "n_used": len(vals),
                "note": "没有任何粗于 fp32 的格式能解释这批 logprob，按 fp32 处理"
                        "（也可能使用了不在检测范围内的量化方案，属于已知盲区）"}
    best = min(hits, key=lambda f: _MANTISSA_BITS[f])
    return {
        "ok": True,
        "precision": best,
        "mantissa_bits": _MANTISSA_BITS[best],
        "all_matching": sorted(hits, key=lambda f: _MANTISSA_BITS[f]),
        "shifts": hits,
        "n_used": len(vals),
        "note": "尾数位越少越可能被量化；int4 等非标准格式会回落成 fp32，属于已知盲区",
    }


def logprob_stats(logprobs: list) -> dict:
    """从 logprob 序列里提取「分布糊不糊」的指标。"""
    top1, ents, margins, n = [], [], [], 0
    for pos in logprobs or []:
        if not isinstance(pos, dict):
            continue
        probs = []
        for a in (pos.get("top_logprobs") or []):
            v = a.get("logprob") if isinstance(a, dict) else None
            if isinstance(v, (int, float)):
                probs.append(math.exp(v))
        if not probs:
            continue
        s = sum(probs) or 1.0
        probs = sorted((p / s for p in probs), reverse=True)
        top1.append(probs[0])
        ents.append(-sum(p * math.log(p + 1e-12) for p in probs))
        if len(probs) >= 2:
            margins.append(probs[0] - probs[1])
        n += 1
    if not n:
        return {"ok": False, "reason": "响应里没有 top_logprobs"}
    return {
        "ok": True,
        "n": n,
        "mean_top1_prob": round(statistics.mean(top1), 4),
        "mean_entropy": round(statistics.mean(ents), 4),
        "mean_margin": round(statistics.mean(margins), 4) if margins else None,
        "min_top1_prob": round(min(top1), 4),
    }

# --------------------------------------------------------------------------
# L2：确定性检查器
# --------------------------------------------------------------------------

def _norm(s: str) -> str:
    return re.sub(r"[,\s_]", "", (s or "").lower())


def check_exact(text: str, params: dict, res: CallResult):
    """回答里必须出现（且基本上只有）这个整数/字符串。"""
    want = params["expect"]
    got = _norm(text)
    if _norm(want) in got:
        extra = len(re.findall(r"\d+", text or "")) - 1
        return (1.0 if extra <= 2 else 0.7), f"命中 {want}；回答里另有 {max(0, extra)} 个数字"
    return 0.0, f"期望 {want}，实际：{(text or '').strip()[:80]!r}"


def check_letters(text: str, params: dict, res: CallResult):
    want = [str(x) for x in params["expect"]]
    got = re.findall(r"\d+", text or "")
    if got[:len(want)] == want:
        return 1.0, f"正确：{','.join(want)}"
    return 0.0, f"期望 {','.join(want)}，实际 {','.join(got[:len(want)]) or '没有数字'}"


def check_json(text: str, params: dict, res: CallResult):
    """必须是严格 JSON（不许有 markdown 围栏），且字段类型正确。"""
    t = (text or "").strip()
    fenced = t.startswith("```")
    body = t
    if fenced:
        m = re.search(r"```[a-zA-Z]*\s*\n([\s\S]*?)```", t)
        body = m.group(1).strip() if m else t
    def check_json_tolerant(text):
        """容忍「复制粘贴时多包了一层引号」这种常见情况。"""
        t = (text or "").strip()
        if len(t) >= 2 and t[0] == '"' and t[-1] == '"':
            t2 = t[1:-1].replace('\\"', '"').replace("\\\\", "\\")
            try:
                return json.loads(t2), True
            except Exception:
                pass
        return json.loads(t), False

    try:
        obj, unquoted = check_json_tolerant(body)
    except Exception as e:
        return 0.0, f"不是合法 JSON：{e}"
    if not isinstance(obj, dict):
        return 0.2, f"顶层不是对象，而是 {type(obj).__name__}"
    score, notes = 1.0, []
    if unquoted:
        notes.append("外层多包了一层引号（粘贴常见，已自动容错）")
    if fenced:
        score -= 0.3
        notes.append("套了 markdown 围栏（严格模式下算失败）")
    for k, typ in params["schema"].items():
        if k not in obj:
            score -= 0.4
            notes.append(f"缺少字段 {k}")
        elif typ == "array":
            if not isinstance(obj[k], list):
                score -= 0.3
                notes.append(f"{k} 不是数组")
            elif params.get("array_len") and len(obj[k]) != params["array_len"]:
                score -= 0.2
                notes.append(f"{k} 长度 {len(obj[k])}，要求 {params['array_len']}")
        elif typ == "int":
            if not isinstance(obj[k], int) or isinstance(obj[k], bool):
                score -= 0.3
                notes.append(f"{k} 不是整数")
        elif typ == "str":
            if not isinstance(obj[k], str):
                score -= 0.3
                notes.append(f"{k} 不是字符串")
    if params.get("forbid_extra") and set(obj) - set(params["schema"]):
        score -= 0.2
        notes.append(f"多了字段 {sorted(set(obj) - set(params['schema']))}")
    return max(0.0, score), "；".join(notes) or "严格 JSON，字段与类型全对"


def check_words(text: str, params: dict, res: CallResult):
    """精确指令遵循：词数 / 结尾词 / 大小写 / 标点。"""
    t = (text or "").strip()
    words = re.findall(r"[A-Za-z']+", t)
    score, notes = 1.0, []
    if len(words) != params["n_words"]:
        score -= 0.5
        notes.append(f"词数 {len(words)}，要求 {params['n_words']}")
    if params.get("last_word") and (not words or words[-1].lower() != params["last_word"]):
        score -= 0.3
        notes.append(f"结尾词是 {words[-1] if words else '空'}，要求 {params['last_word']}")
    if params.get("lowercase") and t != t.lower():
        score -= 0.2
        notes.append("出现了大写")
    if params.get("no_punct") and re.search(r"[.,!?;:，。！？；：]", t):
        score -= 0.2
        notes.append("出现了标点")
    return max(0.0, score), "；".join(notes) or "完全符合约束"


def check_sequence(text: str, params: dict, res: CallResult):
    """1..N 的完整列举：既测长输出能力，也测是否被截断/循环。"""
    n = params["n"]
    nums = [int(x) for x in re.findall(r"\b\d+\b", text or "")]
    present = set(nums)
    missing = [i for i in range(1, n + 1) if i not in present]
    score = 1.0 - min(1.0, len(missing) / n)
    note = f"{n - len(missing)}/{n} 个数字齐全"
    if not missing:
        if res.finish_reason == "length":
            score = 0.8
            note += "，但被 max_tokens 截断"
        else:
            note += "，且没有截断"
    else:
        note += f"，缺 {len(missing)} 个（最大只到 {max(present) if present else 0}）"
    if len(nums) > 3 * n:
        score *= 0.6
        note += "；出现大量重复数字（疑似循环）"
    return max(0.0, score), note


def check_chinese(text: str, params: dict, res: CallResult):
    """中文长文：字数够、不是空话、没有明显退化。"""
    t = (text or "").strip()
    han = re.findall(r"[\u4e00-\u9fff]", t)
    score, notes = 1.0, []
    lo, hi = params.get("min_han", 90), params.get("max_han", 260)
    if len(han) < lo:
        score -= 0.5
        notes.append(f"只有 {len(han)} 个汉字，要求 ≥{lo}")
    elif len(han) > hi:
        score -= 0.15
        notes.append(f"{len(han)} 个汉字，超出 {hi} 上限（没遵守篇幅）")
    if "\ufffd" in t:
        score -= 0.4
        notes.append("出现替换字符（编码坏了）")
    for w in (params.get("must_contain") or []):
        if w not in t:
            score -= 0.15
            notes.append(f"没提到「{w}」")
    if re.search(r"(.{8,})\1", t):
        score -= 0.35
        notes.append("出现连续复读")
    return max(0.0, score), "；".join(notes) or f"{len(han)} 个汉字，通顺无复读"


def check_needle(text: str, params: dict, res: CallResult):
    """长上下文召回。"""
    want = params["needle"]
    if want.lower() in (text or "").lower():
        return 1.0, f"正确召回 {want}"
    return 0.0, f"没有召回 {want}，实际：{(text or '').strip()[:80]!r}"


def check_python(text: str, params: dict, res: CallResult):
    """把模型写的 Python 代码在子进程里真跑一遍（有超时，-I 隔离，不联网）。"""
    code = extract_code(text, lang_hints=("python", "py"))
    if not code or "def " not in code:
        return 0.0, "没有给出可用的 Python 函数"
    harness = code + "\n\n" + params["tests"] + "\n"
    try:
        p = subprocess.run([sys.executable, "-I", "-c", harness],
                           capture_output=True, text=True,
                           timeout=params.get("timeout", 8), cwd="/tmp")
        if p.returncode == 0:
            return 1.0, "单元测试全部通过"
        err = (p.stderr or "").strip().splitlines()
        return 0.0, f"测试失败：{err[-1][:160] if err else p.stdout[:160]}"
    except subprocess.TimeoutExpired:
        return 0.0, "代码超时（疑似死循环）"
    except Exception as e:
        return 0.0, f"执行异常：{e}"


CHECKERS = {
    "exact": check_exact,
    "letters": check_letters,
    "json": check_json,
    "words": check_words,
    "sequence": check_sequence,
    "chinese": check_chinese,
    "needle": check_needle,
    "python": check_python,
}


# --------------------------------------------------------------------------
# 探针定义
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Probe:
    id: str
    group: str                     # visual | capability | fingerprint
    prompt: str
    kind: str = "text"             # svg | html | text
    checker: str = ""              # CHECKERS 里的键；空表示只记录不判分
    params: dict = dataclasses.field(default_factory=dict)
    max_tokens: int = 1024
    temperature: float = 0.0
    system: str = ""
    weight: float = 1.0
    needs_logprobs: bool = False
    runs: int = 1
    desc: str = ""


def build_probes() -> list:
    P = []
    A = P.append

    # ---------------- L1 视觉 / agentic ----------------
    A(Probe("pelican_svg", "visual", "Generate an SVG of a pelican riding a bicycle",
            kind="svg", desc="经典鹈鹕测试：看它能不能把鸟和车真的拼在一起"))
    A(Probe("pelican_animated", "visual",
            "Create a single self-contained HTML file that shows an animated pelican "
            "riding a bicycle, using inline SVG plus CSS animation (the wheels should "
            "spin). Output only the HTML, no explanation.",
            kind="html", max_tokens=6000,
            desc="升级版：动画 HTML，额外考察时序/状态组织能力"))
    A(Probe("capybara_unicycle", "visual", "Generate an SVG of a capybara riding a unicycle",
            kind="svg", params={"scorer": "generic"},
            desc="换一个同样罕见的组合，排除背答案"))
    A(Probe("scene_control", "visual", "Generate an SVG of a house with a door and two windows",
            kind="svg", max_tokens=2000, params={"scorer": "generic"},
            desc="对照组：简单场景。连这个都画不好说明模型已严重受损"))

    # ---------------- L2 确定性能力 ----------------
    A(Probe("arith_mul", "capability",
            "Compute 4783 * 692. Reply with only the final integer, no explanation.",
            checker="exact", params={"expect": "3309836"}, max_tokens=64,
            desc="多位数乘法：量化/小模型最先崩的地方"))
    A(Probe("arith_chain", "capability",
            "Compute 17*23 + 89*11 - 456. Reply with only the final integer, no explanation.",
            checker="exact", params={"expect": "914"}, max_tokens=64,
            desc="混合运算，考察中间步骤不丢"))
    A(Probe("count_letters", "capability",
            "Count how many times the letter 'r' appears in each of these words: "
            "strawberry, raspberry, refrigerator. "
            "Reply with only three integers separated by commas.",
            checker="letters", params={"expect": [3, 3, 4]}, max_tokens=64,
            desc="字母计数：tokenizer 层面的经典坑"))
    A(Probe("json_strict", "capability",
            "Return ONLY a JSON object (no markdown fences, no commentary) with exactly "
            "these keys: name (string), age (integer), tags (array of exactly 3 strings).",
            checker="json",
            params={"schema": {"name": "str", "age": "int", "tags": "array"},
                    "array_len": 3, "forbid_extra": True},
            max_tokens=200, desc="严格 JSON：结构化输出能力"))
    A(Probe("instruction_exact", "capability",
            "Reply with exactly 5 words, all lowercase, no punctuation, and the last word "
            "must be banana.",
            checker="words", params={"n_words": 5, "last_word": "banana",
                                     "lowercase": True, "no_punct": True},
            max_tokens=64, runs=3, desc="精确指令遵循，跑 3 次顺便看温度 0 的稳定性"))
    A(Probe("long_list", "capability",
            "List the integers from 1 to 200, one per line, nothing else.",
            checker="sequence", params={"n": 200}, max_tokens=2000,
            desc="长输出完整性：测截断、测循环"))
    A(Probe("zh_long", "capability",
            "用中文写一段 120 字左右的说明，解释机器学习里的「量化（quantization）」是什么意思，"
            "要提到「精度」和「显存」。只输出正文，不要标题、不要列表。",
            checker="chinese", params={"min_han": 90, "max_han": 260,
                                       "must_contain": ["精度", "显存"]},
            max_tokens=600, desc="中文长文：很多被量化的模型中文先退化"))
    A(Probe("needle", "capability",
            _needle_text("XZ-4471") + "\n\n上面记录里「备用钥匙的编号」是多少？只回答编号。",
            checker="needle", params={"needle": "XZ-4471"}, max_tokens=64,
            desc="长上下文召回（约 4k tokens 的 haystack）"))
    A(Probe("code_fix", "capability",
            "This Python function is buggy:\n\n"
            "def median(nums):\n"
            "    s = sorted(nums)\n"
            "    n = len(s)\n"
            "    return s[n // 2]\n\n"
            "It should return the correct median for both odd and even length lists "
            "(for even length, the average of the two middle values). "
            "Return ONLY the fixed Python function, no explanation, no markdown fences.",
            checker="python", max_tokens=600,
            params={"tests": (
                "assert median([3,1,2]) == 2\n"
                "assert median([1,2,3,4]) == 2.5\n"
                "assert median([5]) == 5\n"
                "assert median([4,4,4,4]) == 4\n"
                "assert median([10,2,8,4]) == 6.0\n"
                "print('ok')\n")},
            desc="修 bug 并真实执行测试：最能区分「会写」和「像在写」"))
    A(Probe("selfid", "capability",
            "Which model and version are you? Answer in one short sentence.",
            weight=0.5, max_tokens=120,
            desc="自报身份。弱信号，但供应商换模型时经常露馅"))

    return P

    desc: str = ""


def _needle_text(needle: str, filler_lines: int = 220) -> str:
    """造一段长 haystack，把 needle 埋在中间。"""
    lines = [f"记录 {i:03d}：仓库 {i} 号的库存为 {(i * 37) % 97} 件，负责人代号 K{i % 11}。"
             for i in range(filler_lines)]
    pos = len(lines) // 2
    lines[pos] = f"记录 {pos:03d}：特别标注 —— 备用钥匙的编号是 {needle}。"
    return "\n".join(lines)


# --------------------------------------------------------------------------
# 从回答里抠代码
# --------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n(.*?)```", re.S)


def extract_code(text: str, lang_hints=("svg", "html", "xml", "python", "py")) -> str:
    """从可能带 markdown 围栏的回答里抠出真正的代码。"""
    if not text:
        return ""
    for m in _FENCE_RE.finditer(text):
        block = m.group(1)
        head = text[max(0, m.start() - 12):m.start()].lower()
        if any(h in head for h in lang_hints) or block.lstrip()[:5].lower() in (
                "<svg ", "<svg>", "<!doc", "<html"):
            return block.strip()
    blocks = _FENCE_RE.findall(text)
    if blocks:
        return max(blocks, key=len).strip()  # 最长的通常才是完整实现
    m = re.search(r"<!DOCTYPE html[\s\S]*", text, re.I)
    if m:
        return m.group(0).strip()
    m = re.search(r"<html[\s\S]*?</html>", text, re.I)
    if m:
        return m.group(0).strip()
    m = re.search(r"<svg[\s\S]*?</svg>", text, re.I)
    if m:
        return m.group(0).strip()
    return text.strip()

SVG_NS = "http://www.w3.org/2000/svg"


# --------------------------------------------------------------------------
# L3：指纹探针（量化 / 换模型检测）
# --------------------------------------------------------------------------

# 这些提示的「下一个 token」几乎是被锁死的，量化后 logprob 会先变糊、再答错
LOGPROB_BATTERY = [
    ("2 + 2 =", "4"),
    ("The capital of France is", "Paris"),
    ("1, 2, 3, 4, 5,", "6"),
    ("The opposite of hot is", "cold"),
    ("The first letter of the alphabet is", "A"),
    ("Roses are red, violets are", "blue"),
    ("The sun rises in the", "east"),
    ("7 times 8 equals", "56"),
    ("The chemical symbol for gold is", "Au"),
    ("Water is made of hydrogen and", "oxygen"),
    ("The largest planet in our solar system is", "Jupiter"),
    ("Python's file extension is dot", "py"),
]

# 固定字符串的 prompt_tokens 由 tokenizer 决定；同一个标称模型在不同供应商处
# 应该给出完全一致的计数。对不上 = 底层不是同一个 tokenizer = 换了模型。
TOKENIZER_STRINGS = [
    "hello world",
    "The quick brown fox jumps over the lazy dog.",
    "中华人民共和国",
    "def f(x): return x**2",
    "1234567890",
]


def run_logprob_battery(target: Target, top_logprobs: int = 20) -> dict:
    """12 个高确定性提示 + max_tokens=1，量测分布糊度并反推 logit 精度。"""
    per_prompt, all_positions, errs = [], [], []
    for prompt, expect in LOGPROB_BATTERY:
        r = call_chat(target, [{"role": "user", "content": prompt}],
                      max_tokens=1, temperature=0.0, want_logprobs=True,
                      top_logprobs=top_logprobs)
        if not r.ok:
            errs.append(f"{prompt!r}: {r.error[:80]}")
            continue
        st = logprob_stats(r.logprobs)
        chosen = ""
        if r.logprobs and isinstance(r.logprobs[0], dict):
            chosen = (r.logprobs[0].get("token") or "").strip()
        hit = _norm(chosen) == _norm(expect) or (chosen and _norm(expect) in _norm(chosen))
        per_prompt.append({
            "prompt": prompt, "expected": expect, "got": chosen,
            "match": bool(hit), "top1_prob": st.get("mean_top1_prob"),
            "entropy": st.get("mean_entropy"),
        })
        all_positions.extend(r.logprobs or [])

    if not per_prompt:
        return {"ok": False, "reason": "logprobs 全部失败", "errors": errs[:3],
                "supported": False}

    agg = logprob_stats(all_positions)
    # 精度反推：挑 top-k 最丰富的位置
    idx = pick_richest_position(all_positions)
    prec = infer_logit_precision(position_logprobs(all_positions, idx))
    matches = sum(1 for p in per_prompt if p["match"])
    return {
        "ok": True, "supported": True,
        "n_prompts": len(per_prompt),
        "top1_agreement": round(matches / len(per_prompt), 3),
        "mean_top1_prob": agg.get("mean_top1_prob"),
        "mean_entropy": agg.get("mean_entropy"),
        "mean_margin": agg.get("mean_margin"),
        "precision": prec,
        "per_prompt": per_prompt,
        "errors": errs[:3],
    }


def run_tokenizer_fp(target: Target) -> dict:
    """用固定字符串的 prompt_tokens 给 tokenizer 打指纹。"""
    counts, errs = {}, []
    for s in TOKENIZER_STRINGS:
        r = call_chat(target, [{"role": "user", "content": s}],
                      max_tokens=1, temperature=0.0)
        if not r.ok:
            errs.append(f"{s!r}: {r.error[:60]}")
            continue
        n = (r.usage or {}).get("prompt_tokens")
        if n is None:
            # 有些网关不返回 usage，退回按字符数估（只能横向比，不能绝对判定）
            n = None
        counts[s] = n
    have = [n for n in counts.values() if n is not None]
    return {
        "ok": bool(have),
        "counts": counts,
        "fingerprint": have or None,
        "note": "同一标称模型在不同供应商处这几个数必须完全一致"
                if have else "网关未返回 usage.prompt_tokens，无法比对",
        "errors": errs[:2],
    }


def run_protocol_probe(target: Target) -> dict:
    """协议与行为指纹：system 遵循、model 回显、usage、温度 0 的确定性。"""
    out = {}

    r = call_chat(target, [
        {"role": "system", "content": "You must answer with exactly one word, in French, and nothing else."},
        {"role": "user", "content": "What colour is a clear daytime sky?"},
    ], max_tokens=32, temperature=0.0)
    if r.ok:
        words = re.findall(r"[A-Za-zÀ-ÿ']+", r.text or "")
        out["system_followed"] = (len(words) == 1)
        out["system_answer"] = (r.text or "").strip()[:40]
    else:
        out["system_followed"] = None
        out["system_answer"] = ""
        out["system_error"] = r.error[:120]

    r2 = call_chat(target, [{"role": "user", "content": "Say the single word: ok"}],
                   max_tokens=8, temperature=0.0)
    out["model_echo"] = r2.model_echo if r2.ok else ""
    out["usage_present"] = bool(r2.ok and (r2.usage or {}).get("total_tokens"))

    # 温度 0 跑 3 次是否一致（非确定性本身也是「后面挂了别的实现」的信号）
    texts = []
    for _ in range(3):
        rr = call_chat(target, [{"role": "user",
                                 "content": "Name a random number between 1 and 1000. Digits only."}],
                       max_tokens=8, temperature=0.0)
        if rr.ok:
            texts.append((rr.text or "").strip())
    out["determinism_samples"] = texts
    out["deterministic"] = (len(set(texts)) == 1) if len(texts) >= 2 else None
    return out


# --------------------------------------------------------------------------
# 运行器
# --------------------------------------------------------------------------

@dataclasses.dataclass
class ProbeRun:
    probe_id: str
    group: str
    kind: str = "text"
    desc: str = ""
    weight: float = 1.0
    ok: bool = False
    score: float = 0.0
    detail: str = ""
    error: str = ""
    text: str = ""
    code: str = ""
    criteria: list = dataclasses.field(default_factory=list)
    analysis: dict = dataclasses.field(default_factory=dict)
    latency_ms: int = 0
    finish_reason: str = ""
    consistency: float | None = None
    scored: bool = True
    extra: dict = dataclasses.field(default_factory=dict)


def _svg_summary(a: SvgAnalysis) -> dict:
    return {
        "xml_valid": a.xml_valid, "n_elements": a.n_elements, "n_shapes": a.n_shapes,
        "n_paths": a.n_paths, "n_groups": a.n_groups, "n_curves": a.n_curves,
        "n_text": a.n_text, "canvas": [a.canvas_w, a.canvas_h],
        "has_viewbox": a.has_viewbox, "distinct_colors": a.distinct_colors,
        "wheels_found": a.wheel_pair is not None,
        "body_found": a.body is not None, "head_found": a.head is not None,
        "beak_found": a.beak is not None, "n_legs": a.n_legs,
        "n_frame_lines": a.n_frame_lines, "chars": a.chars,
        "duplicate_ratio": round(a.duplicate_ratio, 3),
    }


def _new_run(probe: Probe) -> ProbeRun:
    return ProbeRun(probe_id=probe.id, group=probe.group, kind=probe.kind,
                    desc=probe.desc, weight=probe.weight)


def score_answers(probe: Probe, texts: list, exec_code: bool = True,
                  finish_reason: str = "stop") -> ProbeRun:
    """把一批「回答原文」按探针规则打分。**不涉及任何网络调用。**

    在线模式喂 API 返回；离线模式喂人工从 agent 对话框 / 网页里收集来的回答。
    两条路走的是完全一样的评分和报告逻辑——这样即使模型只能通过别的 agent
    或网页访问，你也能得到同样的结论。
    """
    run = _new_run(probe)
    texts = [t for t in (texts or []) if t is not None]
    if not texts:
        run.ok = False
        run.error = "没有拿到任何回答"
        run.detail = "调用失败"
        return run

    run.ok = True
    run.finish_reason = finish_reason
    run.text = texts[0]
    if len(texts) >= 2:
        # 一致性：多数投票占比（在线时是重复调用，离线时是同一题提交多次）
        modal = max(set(texts), key=texts.count)
        run.consistency = round(texts.count(modal) / len(texts), 3)

    # 视觉类：抠代码 → 结构分析 → 打分
    if probe.kind in ("svg", "html"):
        run.code = extract_code(run.text)
        if probe.kind == "svg":
            a = analyze_svg(run.code)
            scorer = (score_svg_generic if probe.params.get("scorer") == "generic"
                      else score_svg)
            s = scorer(a)
            run.analysis = _svg_summary(a)
            run.criteria = s["criteria"]
            run.score = s["score"]
            run.detail = ("结构完整" if run.score > 0.75 else
                          "基本成形" if run.score > 0.5 else
                          "明显缺件" if run.score > 0.3 else "几乎不成形")
        else:
            m = re.search(r"<svg[\s\S]*?</svg>", run.code, re.I)
            run.analysis = _svg_summary(analyze_svg(m.group(0))) if m else {}
            s = score_html(run.code)
            run.criteria = s["criteria"]
            run.score = s["score"]
            run.detail = f"HTML 动画综合 {run.score*100:.0f}"
        if run.consistency is not None:
            run.score *= 0.5 + 0.5 * run.consistency
        return run

    # 文本类：走 checker
    if probe.checker:
        fn = CHECKERS.get(probe.checker)
        if not fn:
            run.score, run.detail = 0.0, f"未知检查器 {probe.checker}"
            return run
        if probe.checker == "python" and not exec_code:
            run.scored = False
            run.score, run.detail = 0.0, "已用 --no-exec 禁用代码执行（仅记录）"
            return run
        scores, details = [], []
        for t in texts:
            sc, dt = fn(t, probe.params, CallResult(ok=True, text=t,
                                                    finish_reason=finish_reason))
            scores.append(sc)
            details.append(dt)
        run.score = sum(scores) / len(scores)
        run.detail = details[0]
        if len(set(details)) > 1:
            run.detail += f"（{len(set(details))} 次结果不同）"
        return run

    # 只记录不判分的探针
    run.scored = False
    run.score = 0.0
    run.detail = (run.text or "").strip()[:200]
    return run


def run_probe(target: Target, probe: Probe, exec_code: bool = True) -> ProbeRun:
    """在线模式：调用 API 再走 score_answers。runs>1 时会重复调用看一致性。"""
    messages = []
    if probe.system:
        messages.append({"role": "system", "content": probe.system})
    messages.append({"role": "user", "content": probe.prompt})

    results = []
    for _ in range(max(1, probe.runs)):
        results.append(call_chat(target, messages, max_tokens=probe.max_tokens,
                                 temperature=probe.temperature,
                                 want_logprobs=probe.needs_logprobs))

    good = [r for r in results if r.ok]
    if not good:
        run = _new_run(probe)
        run.ok = False
        run.error = (results[0].error if results else "没有结果")[:300]
        run.detail = "调用失败"
        return run

    run = score_answers(probe, [r.text or "" for r in good], exec_code=exec_code,
                        finish_reason=good[0].finish_reason)
    run.latency_ms = max((r.latency_ms for r in results), default=0)
    return run


def _save_artifact(tdir: str, run: ProbeRun):
    if not run.ok:
        return
    ext = {"svg": ".svg", "html": ".html"}.get(run.kind, ".txt")
    body = run.code if run.kind in ("svg", "html") and run.code else run.text
    with open(os.path.join(tdir, run.probe_id + ext), "w", encoding="utf-8") as f:
        f.write(body or "")
    if run.kind == "html":
        # 也存一份纯 SVG，方便直接看图
        code = run.code or run.text
        m = re.search(r"<svg[\s\S]*?</svg>", code, re.I)
        if m:
            with open(os.path.join(tdir, run.probe_id + ".svg"), "w", encoding="utf-8") as f:
                f.write(m.group(0))


def run_target(target: Target, probes: list, outdir: str,
               exec_code: bool = True, jobs: int = 1,
               do_fingerprint: bool = True, log=lambda *_a: None) -> dict:
    """把一个 target 上的所有探针跑完，返回原始结果（同时落盘）。"""
    tdir = os.path.join(outdir, target.slug)
    os.makedirs(tdir, exist_ok=True)
    log(f"  [{target.name}] 跑 {len(probes)} 个探针...")

    def work(p):
        return run_probe(target, p, exec_code=exec_code)

    if jobs > 1 and len(probes) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=jobs) as ex:
            runs = list(ex.map(work, probes))
    else:
        runs = [work(p) for p in probes]

    for r in runs:
        _save_artifact(tdir, r)

    fp = {}
    if do_fingerprint:
        log(f"  [{target.name}] 跑 L3 指纹（logprob / tokenizer / 协议）...")
        fp["logprob"] = run_logprob_battery(target)
        fp["tokenizer"] = run_tokenizer_fp(target)
        fp["protocol"] = run_protocol_probe(target)

    return {
        "target": dataclasses.asdict(target) | {"api_key": "***" if target.api_key else ""},
        "slug": target.slug,
        "runs": [dataclasses.asdict(r) for r in runs],
        "fingerprint": fp,
    }


def group_scores(runs: list) -> dict:
    """按 group 汇总加权分数。"""
    out = {}
    for g in ("visual", "capability"):
        rs = [r for r in runs if r["group"] == g and r["ok"] and r.get("scored", True)]
        if not rs:
            out[g] = None
            continue
        wsum = sum(r["weight"] for r in rs)
        out[g] = round(sum(r["score"] * r["weight"] for r in rs) / wsum, 4) if wsum else None
    return out


# --------------------------------------------------------------------------
# 判定引擎：把多路信号合成「量化 / 换模型 / 能力差距」的结论
# --------------------------------------------------------------------------

_PREC_LABEL = {
    "fp32": "FP32（全精度）",
    "fp16": "FP16",
    "bf16": "BF16",
    "fp8_e4m3": "FP8-E4M3（低精度）",
    "fp8_e5m2": "FP8-E5M2（很低精度）",
}


def analyze_fleet(results: list, gap: float = 0.18) -> list:
    """横向对比所有 target，给出可解释的判定。

    单个 target 内部无法判断「好不好」，只有横向对比才有意义：
    同一批探针下，分数明显掉队 / 分布明显变糊 / 指纹对不上的那个，才可疑。
    """
    rows = []
    for res in results:
        gs = group_scores(res["runs"])
        fp = res.get("fingerprint") or {}
        lp = fp.get("logprob") or {}
        prec = (lp.get("precision") or {}).get("precision") if lp.get("ok") else None
        rows.append({
            "slug": res["slug"],
            "name": res["target"].get("name", res["slug"]),
            "model": res["target"].get("model", ""),
            "visual": gs.get("visual"),
            "capability": gs.get("capability"),
            "top1_agreement": lp.get("top1_agreement"),
            "mean_top1_prob": lp.get("mean_top1_prob"),
            "mean_entropy": lp.get("mean_entropy"),
            "precision": prec,
            "tokenizer": (fp.get("tokenizer") or {}).get("fingerprint"),
            "deterministic": (fp.get("protocol") or {}).get("deterministic"),
            "model_echo": (fp.get("protocol") or {}).get("model_echo"),
            "offline": not fp,
            "flags": [],
        })

    def best(key):
        vals = [r[key] for r in rows if r.get(key) is not None]
        return max(vals) if vals else None

    vbest, cbest = best("visual"), best("capability")
    pbest = None
    for r in rows:
        if r["precision"]:
            pbest = r["precision"] if pbest is None else (
                pbest if _MANTISSA_BITS.get(pbest, 23) >= _MANTISSA_BITS.get(r["precision"], 23)
                else r["precision"])

    for r in rows:
        f = r["flags"]
        if r["visual"] is not None and vbest is not None and vbest - r["visual"] >= gap:
            f.append({"sev": "warn", "kind": "visual_gap",
                      "msg": f"视觉分比同批最好低 {(vbest-r['visual'])*100:.0f} 分"})
        if r["capability"] is not None and cbest is not None and cbest - r["capability"] >= gap:
            f.append({"sev": "warn", "kind": "capability_gap",
                      "msg": f"能力分比同批最好低 {(cbest-r['capability'])*100:.0f} 分"})
        if r["top1_agreement"] is not None and r["top1_agreement"] < 0.85:
            f.append({"sev": "warn", "kind": "logprob_disagree",
                      "msg": f"高确定性提示只有 {r['top1_agreement']*100:.0f}% 答对，分布已经不稳"})
        if r["precision"]:
            mb = _MANTISSA_BITS.get(r["precision"], 23)
            if mb <= 3:
                f.append({"sev": "high", "kind": "low_precision",
                          "msg": f"logits 精度反推为 {_PREC_LABEL.get(r['precision'], r['precision'])}，"
                                 "远低于同批其他端点，强烈提示低精度推理 / 量化"})
            elif mb <= 10 and pbest is not None and _MANTISSA_BITS.get(pbest, 23) > mb:
                f.append({"sev": "warn", "kind": "low_precision",
                          "msg": f"logits 精度反推为 {_PREC_LABEL.get(r['precision'], r['precision'])}，"
                                 f"比同批最精细的 {_PREC_LABEL.get(pbest, pbest)} 粗一档"})
            elif mb <= 10:
                f.append({"sev": "info", "kind": "precision_normal",
                          "msg": f"logits 精度反推为 {_PREC_LABEL.get(r['precision'], r['precision'])}"
                                 "（BF16/FP16 是现代推理的常态，本身不构成量化证据）"})
        if r["deterministic"] is False:
            f.append({"sev": "info", "kind": "nondeterministic",
                      "msg": "温度 0 下三次结果不一致（网关缓存/路由或采样参数被改写）"})
        if r.get("offline"):
            f.append({"sev": "info", "kind": "no_fingerprint",
                      "msg": "离线模式（人工收集），拿不到 logprobs，"
                             "L3 量化检测不可用，只能看 L1/L2 的相对差距"})

    # tokenizer 交叉比对：同 model 字段但指纹不同 = 换模型
    by_model = {}
    for r in rows:
        if r["tokenizer"] and r["model"]:
            by_model.setdefault(r["model"], []).append(r)
    for model, group in by_model.items():
        sigs = {tuple(r["tokenizer"]) for r in group}
        if len(sigs) > 1:
            for r in group:
                r["flags"].append({"sev": "high", "kind": "tokenizer_mismatch",
                                   "msg": f"同为 {model}，但 tokenizer 指纹与别的供应商不一致 → 底层很可能不是同一个模型"})
    return rows


def verdict_line(r: dict) -> str:
    highs = [f for f in r["flags"] if f["sev"] == "high"]
    if highs:
        return "⛔ 可疑"
    warns = [f for f in r["flags"] if f["sev"] == "warn"]
    if warns:
        return "⚠️ 有差距"
    if r["visual"] is not None or r["capability"] is not None:
        return "✅ 正常"
    return "— 数据不足"


# --------------------------------------------------------------------------
# 报告输出
# --------------------------------------------------------------------------
def _fmt_pct(x):
    return "—" if x is None else f"{x*100:.0f}"


def write_report(outdir: str, results: list, probes: list) -> dict:
    rows = analyze_fleet(results)
    by_slug = {r["slug"]: r for r in rows}

    os.makedirs(outdir, exist_ok=True)
    with open(os.path.join(outdir, "report.json"), "w", encoding="utf-8") as f:
        json.dump({"generated": _dt.datetime.now().isoformat(timespec="seconds"),
                   "version": __version__, "verdicts": rows, "results": results},
                  f, ensure_ascii=False, indent=2)

    L = []
    L.append("# 模型成色体检报告")
    L.append("")
    L.append(f"生成时间：{_dt.datetime.now():%Y-%m-%d %H:%M:%S}　探针数：{len(probes)}")
    L.append("")
    L.append("## 总览")
    L.append("")
    L.append("| 模型 | L1 视觉 | L2 能力 | logit 精度 | 高确定性答对率 | 判定 |")
    L.append("|---|---|---|---|---|---|")
    for r in sorted(rows, key=lambda x: -(x["visual"] or 0)):
        L.append(f"| {r['name']} | {_fmt_pct(r['visual'])} | {_fmt_pct(r['capability'])} | "
                 f"{_PREC_LABEL.get(r['precision'], '—') if r['precision'] else '不支持 logprobs'} | "
                 f"{_fmt_pct(r['top1_agreement'])} | {verdict_line(r)} |")
    L.append("")
    L.append("> 分数都是**横向相对值**：只有把同一批探针同时跑在多个供应商上才有意义。")
    L.append("")

    for r in sorted(rows, key=lambda x: -(x["visual"] or 0)):
        res = next((x for x in results if x["slug"] == r["slug"]), None)
        L.append(f"## {r['name']}")
        L.append("")
        if r["model"]:
            L.append(f"模型字段：`{r['model']}`　回显：`{r['model_echo'] or '—'}`")
        if r["flags"]:
            for fl in r["flags"]:
                mark = {"high": "⛔", "warn": "⚠️", "info": "ℹ️"}[fl["sev"]]
                L.append(f"- {mark} {fl['msg']}")
        else:
            L.append("- ✅ 没有检出异常信号")
        L.append("")
        # 探针明细
        L.append("| 探针 | 组 | 分数 | 说明 |")
        L.append("|---|---|---|---|")
        for run in (res or {}).get("runs", []):
            if not run["ok"]:
                sc = "✗"
            elif not run.get("scored", True):
                sc = "仅记录"
            else:
                sc = f"{run['score']*100:.0f}"
            det = (run.get("error") or run.get("detail") or "").replace("|", "/")[:120]
            L.append(f"| `{run['probe_id']}` | {run['group']} | {sc} | {det} |")
        L.append("")
        # L3 细节
        fp = (res or {}).get("fingerprint") or {}
        lp = fp.get("logprob") or {}
        if lp.get("ok"):
            L.append("**L3 分布指纹**")
            L.append("")
            L.append(f"- 高确定性提示答对率：{_fmt_pct(lp.get('top1_agreement'))}%")
            L.append(f"- 首 token top-1 平均概率：{lp.get('mean_top1_prob')}（越高越自信）")
            L.append(f"- 首 token 平均熵：{lp.get('mean_entropy')}（越低越确定）")
            pr = lp.get("precision") or {}
            if pr.get("ok"):
                L.append(f"- logit 精度反推：**{_PREC_LABEL.get(pr['precision'], pr['precision'])}**"
                         f"（尾数 {pr['mantissa_bits']} 位；可匹配 {', '.join(pr['all_matching'])}）")
            else:
                L.append(f"- logit 精度反推：无法判定（{pr.get('reason')}）")
            L.append("")
        elif fp:
            L.append(f"**L3 分布指纹**：该网关不支持 logprobs（{lp.get('reason','未返回')}），"
                     "量化检测能力受限，只能依赖 L1/L2 与 tokenizer 指纹。")
            L.append("")
        tok = fp.get("tokenizer") or {}
        if tok.get("fingerprint"):
            L.append(f"tokenizer 指纹（prompt_tokens）：{tok['fingerprint']}")
            L.append("")

    L.append("---")
    L.append("")
    L.append("## 判据说明（为什么可以这样判）")
    L.append("")
    for line in JUDGE_NOTES:
        L.append(f"- {line}")
    L.append("")

    with open(os.path.join(outdir, "report.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(L))

    write_gallery(outdir, results, probes)
    return {"rows": rows, "md": os.path.join(outdir, "report.md")}


JUDGE_NOTES = [
    "鹈鹕骑自行车这个组合天然稀缺（训练集里几乎没有现成图），模型没法「背答案」，"
    "只能真的把「鸟坐在车架上、脚够到踏板、翅膀扶车把」的空间关系推出来。",
    "视觉探针的自动打分只看结构：轮子是否是两个尺寸接近的圆、车架是否存在、"
    "鹈鹕的头/喙/腿是否落在正确位置。它抓得住「缺件」，抓不住审美，所以 gallery.html 里的人工看图仍是最终裁判。",
    "多位数乘法、字母计数、严格 JSON、精确词数这类任务几乎不需要「聪明」，"
    "但需要完整的数值/符号通路。量化和小模型最先崩的就是这些。",
    "logprob 是唯一能黑盒量测「分布糊度」的信号：高确定性提示下，"
    "好的模型 top-1 概率接近 1；被量化后分布变平、熵上升、top-1 概率下降。",
    "logit 精度反推（ICLR Blogposts 2026）利用 log-softmax 只做整体平移这一性质："
    "若原始 logits 是低精度存储的，必然存在唯一平移量使全部 logprob 在低精度格式里精确可表示。",
    "tokenizer 指纹是抓「换模型」最硬的一条：同一个标称模型在不同供应商处，"
    "固定字符串的 prompt_tokens 必须完全一致，对不上就说明底层实现不同。",
    "所有这些都是「相对」判断：单次结果没有意义，必须固定模型 ID、参数、"
    "提示词，多供应商同时跑，并且保存 baseline 以便日后复测漂移。",
]


def _sanitize(html: str, keep_svg_only: bool = False) -> str:
    """报告里内联渲染模型输出前，先把脚本和事件属性去掉。"""
    h = re.sub(r"<script[\s\S]*?</script>", "", html or "", flags=re.I)
    h = re.sub(r"\son[a-z]+\s*=\s*(\"[^\"]*\"|'[^']*')", "", h, flags=re.I)
    h = re.sub(r"javascript:", "", h, flags=re.I)
    if keep_svg_only:
        m = re.search(r"<svg[\s\S]*?</svg>", h, re.I)
        h = m.group(0) if m else ""
    return h


def write_gallery(outdir: str, results: list, probes: list):
    """生成 gallery.html：同一个探针下所有模型并排，直接看图比高低。"""
    visual = [p for p in probes if p.kind in ("svg", "html")]
    if not visual:
        return
    parts = ["<!DOCTYPE html><html lang=zh><head><meta charset=utf-8>",
             "<title>modelcheck 视觉对比</title><style>",
             "body{font:14px/1.5 -apple-system,'PingFang SC',sans-serif;margin:24px;background:#fafafa;color:#222}",
             "h1{font-size:20px}h2{margin-top:32px;border-top:1px solid #ddd;padding-top:16px}",
             ".row{display:flex;flex-wrap:wrap;gap:16px}",
             ".card{background:#fff;border:1px solid #e3e3e3;border-radius:10px;padding:10px;width:340px}",
             ".card h3{font-size:13px;margin:0 0 6px}",
             ".box{background:#fff;border:1px dashed #ddd;border-radius:6px;height:230px;display:flex;"
             "align-items:center;justify-content:center;overflow:hidden}",
             ".box svg{max-width:100%;max-height:220px;height:auto}",
             ".score{font-weight:700}.bad{color:#b00}.good{color:#080}",
             ".det{color:#666;font-size:12px;margin-top:6px}",
             "iframe{width:100%;height:230px;border:0;background:#fff}",
             ".prompt{color:#555;font-size:12px;font-family:ui-monospace,monospace;background:#f2f2f2;padding:6px 8px;border-radius:6px}",
             "</style></head><body>",
             f"<h1>modelcheck 视觉对比 · {_dt.datetime.now():%Y-%m-%d %H:%M}</h1>",
             "<p>下面是同一提示词下各模型的原始输出。自动分只是结构分的参考，"
             "最终请自己看图：<b>鹈鹕是否真的骑在车上</b>、轮子是不是轮子、有没有环境与细节。</p>"]

    for p in visual:
        parts.append(f"<h2>{p.id}</h2>")
        parts.append(f"<div class=prompt>{_esc(p.prompt[:300])}</div><div class=row>")
        for res in results:
            run = next((r for r in res["runs"] if r["probe_id"] == p.id), None)
            if not run:
                continue
            name = res["target"].get("name", res["slug"])
            sc = run["score"] * 100
            cls = "good" if sc >= 75 else ("bad" if sc < 45 else "")
            parts.append("<div class=card>")
            parts.append(f"<h3>{_esc(name)}　<span class='score {cls}'>{sc:.0f}</span>/100</h3>")
            if not run["ok"]:
                parts.append(f"<div class=box style='color:#b00'>调用失败：{_esc(run.get('error','')[:120])}</div>")
            elif p.kind == "svg":
                svg = _sanitize(run.get("code") or run.get("text") or "", keep_svg_only=True)
                parts.append(f"<div class=box>{svg or '<span style=color:#999>没有有效 SVG</span>'}</div>")
            else:
                html = _sanitize(run.get("code") or run.get("text") or "")
                svg = _sanitize(html, keep_svg_only=True)
                parts.append(f"<div class=box>{svg or '<span style=color:#999>没有有效 SVG</span>'}</div>")
            parts.append(f"<div class=det>{_esc(run.get('detail') or run.get('error') or '')[:160]}</div>")
            parts.append("</div>")
        parts.append("</div>")
    parts.append("</body></html>")
    with open(os.path.join(outdir, "gallery.html"), "w", encoding="utf-8") as f:
        f.write("\n".join(parts))


def _esc(s: str) -> str:
    return (str(s or "").replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


# --------------------------------------------------------------------------
# 自检用的本地 mock 网关（不需要任何 API key，就能验证整条流水线）
# --------------------------------------------------------------------------

_GOOD_SVG = '''<svg width="400" height="300" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
  <circle cx="270" cy="210" r="45" fill="none" stroke="#333" stroke-width="4"/>
  <circle cx="130" cy="210" r="45" fill="none" stroke="#333" stroke-width="4"/>
  <line x1="130" y1="210" x2="270" y2="210" stroke="#333" stroke-width="4"/>
  <line x1="200" y1="210" x2="200" y2="150" stroke="#333" stroke-width="4"/>
  <line x1="200" y1="150" x2="270" y2="210" stroke="#333" stroke-width="4"/>
  <line x1="200" y1="150" x2="165" y2="125" stroke="#333" stroke-width="4"/>
  <ellipse cx="195" cy="115" rx="42" ry="30" fill="#f7f7f7"/>
  <path d="M195 115 Q215 95 228 82" fill="none" stroke="#f7f7f7" stroke-width="12"/>
  <circle cx="232" cy="76" r="14" fill="#f7f7f7"/>
  <path d="M232 76 Q262 74 274 84 L232 92 Z" fill="orange"/>
  <path d="M195 115 Q225 128 238 152" fill="none" stroke="#f7f7f7" stroke-width="10"/>
  <line x1="185" y1="140" x2="200" y2="152" stroke="orange" stroke-width="4"/>
  <line x1="207" y1="140" x2="200" y2="152" stroke="orange" stroke-width="4"/>
  <circle cx="236" cy="72" r="2.5" fill="#111"/>
</svg>'''

_BAD_SVG = '''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100">
  <rect x="40" y="30" width="20" height="40" fill="black"/>
  <circle cx="45" cy="70" r="4" fill="gray"/>
  <circle cx="55" cy="70" r="4" fill="gray"/>
  <text x="10" y="95">pelican on a bicycle</text>
</svg>'''


_HOUSE = '''<svg width="400" height="300" viewBox="0 0 400 300" xmlns="http://www.w3.org/2000/svg">
  <rect x="120" y="140" width="160" height="120" fill="#e8dcc8" stroke="#8a7a66" stroke-width="4"/>
  <polygon points="120,140 200,80 280,140" fill="#b04a3a"/>
  <rect x="178" y="192" width="48" height="68" fill="#7a5230"/>
  <circle cx="214" cy="228" r="4" fill="#f0d060"/>
  <rect x="138" y="164" width="36" height="36" fill="#a8d8f0" stroke="#8a7a66" stroke-width="3"/>
  <rect x="226" y="164" width="36" height="36" fill="#a8d8f0" stroke="#8a7a66" stroke-width="3"/>
  <line x1="156" y1="164" x2="156" y2="200" stroke="#8a7a66" stroke-width="3"/>
  <line x1="138" y1="182" x2="174" y2="182" stroke="#8a7a66" stroke-width="3"/>
  <line x1="244" y1="164" x2="244" y2="200" stroke="#8a7a66" stroke-width="3"/>
  <line x1="226" y1="182" x2="262" y2="182" stroke="#8a7a66" stroke-width="3"/>
</svg>'''

# 这些提示词的「正确答案」，mock 网关据此模拟正常 / 退化的模型
_MOCK_EXPECT = {
    "2 + 2 =": "4",
    "The capital of France is": "Paris",
    "1, 2, 3, 4, 5,": "6",
    "The opposite of hot is": "cold",
    "The first letter of the alphabet is": "A",
    "Roses are red, violets are": "blue",
    "The sun rises in the": "east",
    "7 times 8 equals": "56",
    "The chemical symbol for gold is": "Au",
    "Water is made of hydrogen and": "oxygen",
    "The largest planet in our solar system is": "Jupiter",
    "Python's file extension is dot": "py",
}


def _mock_top_logprobs(prec: str, seed: int = 3, n: int = 20) -> list:
    """造一批「已知精度」的 logprob，用来验证精度反推。"""
    rnd = random.Random(seed)
    logits = [_nearest(rnd.uniform(-4, 26), prec) for _ in range(n)]
    mx = max(logits)
    lse = mx + math.log(sum(math.exp(v - mx) for v in logits))
    lps = sorted({_f32(v - lse) for v in logits}, reverse=True)
    return [{"token": f"tok{i}", "logprob": lp} for i, lp in enumerate(lps)]


class _MockHandler(__import__("http.server", fromlist=["BaseHTTPRequestHandler"]).BaseHTTPRequestHandler):
    quality = "good"
    prec = "bf16"

    def log_message(self, *a):
        pass

    def do_POST(self):
        n = int(self.headers.get("content-length", 0) or 0)
        try:
            req = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            req = {}
        msgs = req.get("messages") or []
        user = (msgs[-1].get("content") or "") if msgs else ""
        system = " ".join(m.get("content", "") for m in msgs if m.get("role") == "system")
        low = user.lower()
        degraded = self.quality != "good"
        text = "ok"
        logprobs = None
        finish = "stop"

        if req.get("logprobs"):
            lps = _mock_top_logprobs(self.prec)
            tok = lps[0]["token"] if degraded else _MOCK_EXPECT.get(user.strip(), lps[0]["token"])
            logprobs = {"content": [{"token": tok, "logprob": lps[0]["logprob"],
                                     "top_logprobs": lps}]}
            text = tok
        elif "self-contained html" in low:
            if degraded:
                text = "<!DOCTYPE html>\n<html><body>\n" + _BAD_SVG + "\n</body></html>"
            else:
                text = ("<!DOCTYPE html>\n<html><head><meta charset='utf-8'><style>\n"
                        "@keyframes spin{to{transform:rotate(360deg)}}\n"
                        ".wheel{transform-box:fill-box;transform-origin:center;"
                        "animation:spin 2s linear infinite}\n"
                        "</style></head><body>\n"
                        + _GOOD_SVG.replace('stroke="#333" stroke-width="4"',
                                            'class="wheel" stroke="#333" stroke-width="4"')
                        + "\n</body></html>")
        elif "pelican" in low and "svg" in low:
            text = "```svg\n" + (_BAD_SVG if degraded else _GOOD_SVG) + "\n```"
        elif "capybara" in low:
            text = "```svg\n" + (_BAD_SVG if degraded else _GOOD_SVG).replace("pelican", "capybara") + "\n```"
        elif "house" in low:
            text = "```svg\n" + _HOUSE + "\n```"
        elif "4783 * 692" in low:
            text = "3310636" if degraded else "3309836"
        elif "17*23" in low:
            text = "914"
        elif "letter 'r'" in low:
            text = "2, 3, 4" if degraded else "3, 3, 4"
        elif "return only a json object" in low:
            text = '{"name": "x", "age": 3, "tags": ["a","b"]}' if degraded else \
                   '{"name": "x", "age": 3, "tags": ["a","b","c"]}'
        elif "exactly 5 words" in low:
            text = "I like yellow banana" if degraded else "please pass the ripe banana"
        elif "integers from 1 to 200" in low:
            text = "\n".join(str(i) for i in range(1, 201))
            if degraded:
                text = "\n".join(str(i) for i in range(1, 121))
                finish = "length"
        elif "\u91cf\u5316" in user:
            if degraded:
                text = "\u91cf\u5316\u5c31\u662f\u538b\u7f29\u3002"
            else:
                text = ("\u91cf\u5316\u662f\u6307\u628a\u795e\u7ecf\u7f51\u7edc\u91cc\u7684\u53c2\u6570"
                        "\u4ece\u9ad8\u7cbe\u5ea6\u7684\u6d6e\u70b9\u6570\uff08\u4f8b\u5982 FP16\uff09"
                        "\u8f6c\u6362\u6210\u4f4e\u4f4d\u6570\u7684\u6574\u6570\uff0c\u4ece\u800c\u5728"
                        "\u727a\u7272\u4e00\u5b9a\u7cbe\u5ea6\u7684\u524d\u63d0\u4e0b\uff0c\u628a"
                        "\u6a21\u578b\u7684\u663e\u5b58\u5360\u7528\u964d\u5230\u539f\u6765\u7684"
                        "\u56db\u5206\u4e4b\u4e00\u751a\u81f3\u66f4\u4f4e\uff0c\u540c\u65f6\u63d0\u5347"
                        "\u63a8\u7406\u901f\u5ea6\u3002\u5e38\u89c1\u7684\u505a\u6cd5\u5305\u62ec "
                        "INT8 \u4e0e INT4\uff0c\u4ee5\u53ca\u8fd1\u5e74\u6d41\u884c\u7684 FP8\u3002"
                        "\u9700\u8981\u6ce8\u610f\u7684\u662f\uff0c\u5e45\u5ea6\u8fc7\u5927\u7684"
                        "\u91cf\u5316\u4f1a\u8ba9\u6a21\u578b\u5728\u6570\u503c\u63a8\u7406\u548c"
                        "\u957f\u6587\u672c\u4efb\u52a1\u4e0a\u660e\u663e\u9000\u5316\u3002")
        elif "\u5907\u7528\u94a5\u5319" in user:
            text = "XZ-0000" if degraded else "XZ-4471"
        elif "median" in low:
            text = "def median(nums):\n    s = sorted(nums)\n    return s[len(s)//2]" if degraded else \
                   "def median(nums):\n    s = sorted(nums)\n    n = len(s)\n    if n % 2:\n        return s[n//2]\n    return (s[n//2 - 1] + s[n//2]) / 2"
        elif "which model" in low:
            text = "I am a large language model." if degraded else "I am Claude 3.5 Sonnet."
        elif system and "french" in system.lower():
            text = "bleu" if not degraded else "The sky is blue"

        body = {
            "id": "mock", "object": "chat.completion", "model": req.get("model", "mock"),
            "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                         "finish_reason": finish, **(({"logprobs": logprobs}) if logprobs else {})}],
            "usage": {"prompt_tokens": max(1, len(user) // 4) + (0 if not degraded else 1),
                      "completion_tokens": max(1, len(text) // 4),
                      "total_tokens": max(2, (len(user) + len(text)) // 4)},
        }
        raw = json.dumps(body).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)


def start_mock_gateway(quality: str = "good", prec: str = "bf16"):
    """起一个本地 OpenAI 兼容 mock 网关，返回 (base_url, shutdown_fn)。"""
    import http.server
    import threading

    handler = type("H", (_MockHandler,), {"quality": quality, "prec": prec})
    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()

    def shutdown():
        srv.shutdown()
        srv.server_close()

    return f"http://127.0.0.1:{srv.server_address[1]}/v1", shutdown


# --------------------------------------------------------------------------
# 漂移基线：同一个端点，今天 vs 上次
# --------------------------------------------------------------------------

def baseline_snapshot(results: list) -> dict:
    """把这次的关键指标存成基线，供日后复测比较。"""
    snap = {"generated": _dt.datetime.now().isoformat(timespec="seconds"), "targets": {}}
    for res in results:
        lp = (res.get("fingerprint") or {}).get("logprob") or {}
        snap["targets"][res["slug"]] = {
            "model": res["target"].get("model", ""),
            "tokenizer": ((res.get("fingerprint") or {}).get("tokenizer") or {}).get("fingerprint"),
            "mean_top1_prob": lp.get("mean_top1_prob"),
            "mean_entropy": lp.get("mean_entropy"),
            "top1_agreement": lp.get("top1_agreement"),
            "precision": (lp.get("precision") or {}).get("precision") if lp.get("ok") else None,
            "per_prompt_logprob": {
                p["prompt"]: p.get("top1_prob") for p in (lp.get("per_prompt") or [])
            },
        }
    return snap


def compare_baseline(results: list, baseline: dict, tol: float = 0.05) -> list:
    """拿最新结果和基线比：tokenizer 变了 / 分布明显移动 = 端点后面被换过。

    对应 arXiv:2512.03816（Log Probability Tracking of LLM APIs）的思路：
    固定提示词，看首 token 的 logprob 是否发生系统性位移。
    """
    flags = []
    for res in results:
        old = (baseline.get("targets") or {}).get(res["slug"])
        if not old:
            flags.append({"slug": res["slug"], "sev": "info", "msg": "基线里没有这个端点，已新建"})
            continue
        new_tok = ((res.get("fingerprint") or {}).get("tokenizer") or {}).get("fingerprint")
        if old.get("tokenizer") and new_tok and old["tokenizer"] != new_tok:
            flags.append({"slug": res["slug"], "sev": "high",
                          "msg": f"tokenizer 指纹变了：{old['tokenizer']} -> {new_tok}，端点后面的模型被换过"})
        lp = (res.get("fingerprint") or {}).get("logprob") or {}
        old_pp = old.get("per_prompt_logprob") or {}
        moved = []
        for p in (lp.get("per_prompt") or []):
            o = old_pp.get(p["prompt"])
            n = p.get("top1_prob")
            if o is not None and n is not None and abs(n - o) > tol:
                moved.append((p["prompt"], o, n))
        if moved:
            sev = "high" if len(moved) >= 3 else "warn"
            flags.append({"slug": res["slug"], "sev": sev,
                          "msg": f"{len(moved)} 个提示词的首 token logprob 离开基线超过 {tol}，"
                                 f"例如 {moved[0][0]!r}: {moved[0][1]:.3f} -> {moved[0][2]:.3f}"})
        if not any(f["slug"] == res["slug"] for f in flags):
            flags.append({"slug": res["slug"], "sev": "info", "msg": "与基线一致，未检出漂移"})
    return flags


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

# --------------------------------------------------------------------------
# 离线模式：模型在别的 agent / 网页里，没有 API 时怎么办
# --------------------------------------------------------------------------

_EXT_FOR = {"svg": ".svg", "html": ".html", "text": ".txt"}


def emit_prompts(outdir: str, probes: list, log=lambda *_a: None) -> str:
    """把探针题目导出成「可直接粘贴」的清单，供任何 agent / 网页对话框使用。

    生成：
      <outdir>/PROMPTS.md          人看的清单 + 保存说明
      <outdir>/prompts.json        机器可读（想写脚本自动投喂 agent 用这个）
      <outdir>/prompts/<id>.txt    每题一个文件，内容就是原文，复制即用
    """
    pdir = os.path.join(outdir, "prompts")
    os.makedirs(pdir, exist_ok=True)

    lines = ["# modelcheck 离线题目清单", "",
             "模型只能通过别的 agent 或网页访问时，用这个模式：", "",
             "1. 把下面每道题**原文**发给模型（一题一个全新对话，别带上下文，" +
             "否则「精确指令遵循」这类题就不准了）。",
             "2. 把模型的回答**原样**存成文件（连 markdown 围栏一起存也没关系，会自动抠代码）。",
             "3. 目录按下面的约定放，然后跑：",
             "",
             "```bash",
             "python3 modelcheck.py --import-dir answers/ --out runs/manual",
             "```",
             "",
             "## 目录约定",
             "",
             "```",
             "answers/",
             "  <模型别名>/",
             "    pelican_svg.svg            # 图形类：存模型给的整段回答也行",
             "    pelican_animated.html",
             "    arith_mul.txt",
             "    instruction_exact/          # 需要重复的题，放多个文件看一致性",
             "      1.txt",
             "      2.txt",
             "      3.txt",
             "```",
             "",
             "也支持平铺写法：`answers/<模型别名>__<题目id>.txt`。",
             "",
             "> ⚠️ 离线模式拿不到 `logprobs`，所以 L3 的「分布糊度 / logit 精度反推 / " +
             "tokenizer 指纹 / 漂移基线」全部不可用，只剩 L1 视觉 + L2 能力。" +
             "想测量化，优先用下面「情况 A/B」拿 API。",
             "",
             "---", ""]

    for p in probes:
        body = p.prompt
        fn = os.path.join(pdir, p.id + ".txt")
        with open(fn, "w", encoding="utf-8") as f:
            f.write(body)
        times = f"　（这题建议独立跑 {p.runs} 次，看一致性）" if p.runs > 1 else ""
        lines.append(f"## {p.id}　`{p.group}`{times}")
        lines.append("")
        lines.append(f"*{p.desc}*")
        lines.append("")
        lines.append("要保存成：`<模型别名>/" + p.id + _EXT_FOR.get(p.kind, ".txt") + "`"
                     if p.runs <= 1 else
                     "要保存成：`<模型别名>/" + p.id + "/1.txt`（依次 2、3…）")
        lines.append("")
        lines.append("```text")
        lines.append(body if len(body) < 1200 else body[:1200] + "\n...(见 prompts/" + p.id + ".txt)")
        lines.append("```")
        lines.append("")

    md = os.path.join(outdir, "PROMPTS.md")
    with open(md, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    with open(os.path.join(outdir, "prompts.json"), "w", encoding="utf-8") as f:
        json.dump([{"id": p.id, "group": p.group, "kind": p.kind, "runs": p.runs,
                    "desc": p.desc, "prompt": p.prompt,
                    "save_as": p.id + _EXT_FOR.get(p.kind, ".txt")} for p in probes],
                  f, ensure_ascii=False, indent=2)
    log(f"已导出 {len(probes)} 道题 -> {md}")
    return md


def _read_any(path: str) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except OSError:
        return ""


def load_manual_answers(dirpath: str, probes: list, exec_code: bool = True,
                        log=lambda *_a: None) -> list:
    """从人工收集的回答目录生成结果（不联网），产出与在线模式完全同构。"""
    if not os.path.isdir(dirpath):
        raise SystemExit(f"--import-dir 不是目录：{dirpath}")

    by_id = {p.id: p for p in probes}
    # 1) 收集 <target>/<probe>... ，同时支持平铺的 <target>__<probe>
    collected: dict = {}
    for entry in sorted(os.listdir(dirpath)):
        full = os.path.join(dirpath, entry)
        if os.path.isdir(full):
            target = entry
            for sub in sorted(os.listdir(full)):
                subfull = os.path.join(full, sub)
                stem = os.path.splitext(sub)[0]
                if os.path.isdir(subfull):
                    if sub in by_id:
                        texts = [_read_any(os.path.join(subfull, x))
                                 for x in sorted(os.listdir(subfull))]
                        collected.setdefault(target, {})[sub] = texts
                elif stem in by_id:
                    collected.setdefault(target, {})[stem] = [_read_any(subfull)]
        elif os.path.isfile(full):
            stem = os.path.splitext(entry)[0]
            if "__" in stem:
                target, _, pid = stem.partition("__")
                if pid in by_id:
                    collected.setdefault(target, {})[pid] = [_read_any(full)]

    if not collected:
        raise SystemExit(
            f"在 {dirpath} 里没找到任何符合约定的回答文件。\n"
            "期望结构：answers/<模型别名>/<题目id>.txt|.svg|.html\n"
            "先跑 python3 modelcheck.py --emit-prompts prompts/ 看题目清单。")

    results = []
    used_slugs = {}
    for target, answers in sorted(collected.items()):
        runs = []
        for p in probes:
            texts = answers.get(p.id)
            if texts:
                run = score_answers(p, texts, exec_code=exec_code)
            else:
                run = _new_run(p)
                run.ok = False
                run.error = "未提交回答"
                run.detail = "未提交回答（跳过）"
            runs.append(dataclasses.asdict(run))
        got = sum(1 for r in runs if r["ok"])
        log(f"  [{target}] 收到 {got}/{len(probes)} 题的作答")
        slug = Target(name=target, base_url="manual").slug
        used_slugs[slug] = used_slugs.get(slug, 0) + 1
        if used_slugs[slug] > 1:
            slug = f"{slug}-{used_slugs[slug]}"     # 防撞：不同目标不能共用一个 slug
        results.append({
            "target": {"name": target, "model": target, "base_url": "(人工收集)",
                       "api_key": "", "note": "离线模式：无 logprobs", "extra_body": {},
                       "headers": {}},
            "slug": slug,
            "runs": runs,
            "fingerprint": {},          # 离线模式没有 L3
        })
    return results



def load_targets(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    items = data.get("targets", data) if isinstance(data, dict) else data
    out = []
    for it in items:
        key = it.get("api_key", "")
        if isinstance(key, str) and key.startswith("env:"):
            key = os.environ.get(key[4:], "")
        out.append(Target(name=it["name"], base_url=it["base_url"], api_key=key or "",
                          model=it.get("model", ""), note=it.get("note", ""),
                          extra_body=it.get("extra_body") or {},
                          headers=it.get("headers") or {}))
    return out


def selftest() -> int:
    """不需要任何 API key 的自检：打分器、精度反推、检查器、端到端流水线。"""
    ok = True
    print("=" * 68)
    print("modelcheck 自检")
    print("=" * 68)

    # 1) SVG 打分器：官方样例里「好」的必须明显高于「差」的
    fx = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")
    scores = {}
    if os.path.isdir(fx):
        for fn in sorted(os.listdir(fx)):
            if fn.endswith(".svg"):
                with open(os.path.join(fx, fn), encoding="utf-8") as f:
                    scores[fn] = score_svg(analyze_svg(f.read()))["score"]
    if scores:
        best = max(scores, key=scores.get)
        worst = min(scores, key=scores.get)
        print(f"\n[1] SVG 打分器（{len(scores)} 个官方样例）")
        for k, v in sorted(scores.items(), key=lambda kv: -kv[1]):
            print(f"    {v*100:5.1f}  {k}")
        good = scores.get("claude-3-5-sonnet-20241022.svg", 0)
        bad = scores.get("gpt-3.5-turbo.svg", 0)
        passed = good - bad >= 0.15
        ok &= passed
        print(f"    区分度检查：{best} {scores[best]*100:.0f} vs {worst} {scores[worst]*100:.0f} "
              f"-> {'通过' if passed else '失败'}")

    # 2) logit 精度反推
    print("\n[2] logit 精度反推")
    good = 0
    for fmt in ["fp8_e5m2", "fp8_e4m3", "bf16", "fp16", "fp32"]:
        rnd = random.Random(5)
        logits = [_nearest(rnd.uniform(-5, 25), fmt) for _ in range(20)]
        mx = max(logits)
        lse = mx + math.log(sum(math.exp(v - mx) for v in logits))
        vals = sorted({_f32(v - lse) for v in logits}, reverse=True)
        got = infer_logit_precision(vals).get("precision")
        hit = got == fmt
        good += hit
        print(f"    真值 {fmt:10s} -> 判定 {got:10s} {'✓' if hit else '✗'}")
    fp_count = 0
    for seed in range(30):
        rnd = random.Random(900 + seed)
        logits = [rnd.uniform(-8, 30) for _ in range(20)]
        mx = max(logits)
        lse = mx + math.log(sum(math.exp(v - mx) for v in logits))
        vals = sorted({_f32(v - lse) for v in logits}, reverse=True)
        if infer_logit_precision(vals).get("precision") != "fp32":
            fp_count += 1
    passed = good == 5 and fp_count == 0
    ok &= passed
    print(f"    命中 {good}/5，fp32 误报 {fp_count}/30 -> {'通过' if passed else '失败'}")

    # 3) 检查器
    print("\n[3] 确定性检查器")
    R = CallResult(ok=True, text="3309836", finish_reason="stop")
    cases = [
        ("exact", "3309836", {"expect": "3309836"}, 1.0),
        ("exact", "3309835", {"expect": "3309836"}, 0.0),
        ("letters", "3, 3, 4", {"expect": [3, 3, 4]}, 1.0),
        ("json", '{"name":"x","age":3,"tags":["a","b","c"]}',
         {"schema": {"name": "str", "age": "int", "tags": "array"}, "array_len": 3}, 1.0),
        ("words", "please pass the ripe banana",
         {"n_words": 5, "last_word": "banana", "lowercase": True, "no_punct": True}, 1.0),
        ("needle", "XZ-4471", {"needle": "XZ-4471"}, 1.0),
    ]
    cok = True
    for name, text, params, want in cases:
        got, _ = CHECKERS[name](text, params, CallResult(ok=True, text=text))
        hit = abs(got - want) < 1e-6
        cok &= hit
        print(f"    {name:8s} 期望 {want:4.1f} 得到 {got:4.1f} {'✓' if hit else '✗'}")
    code = ("def median(nums):\n    s = sorted(nums)\n    n = len(s)\n"
            "    if n % 2: return s[n//2]\n    return (s[n//2-1]+s[n//2])/2")
    got, det = CHECKERS["python"](code, {"tests": "assert median([1,2,3,4])==2.5\nprint('ok')\n"},
                                  CallResult(ok=True, text=code))
    print(f"    python   期望 1.0 得到 {got:4.1f} ({det}) {'✓' if got == 1.0 else '✗'}")
    ok &= cok and got == 1.0

    # 4) 端到端：两个 mock 网关，一个正常一个「被量化」
    print("\n[4] 端到端流水线（本地 mock 网关，无需 API key）")
    url_good, stop_g = start_mock_gateway("good", "bf16")
    url_bad, stop_b = start_mock_gateway("degraded", "fp8_e4m3")
    try:
        targets = [Target("mock-good", url_good, "k", "mock-llm-1"),
                   Target("mock-degraded", url_bad, "k", "mock-llm-2")]
        probes = build_probes()
        outdir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "selftest-run")
        results = [run_target(t, probes, outdir, jobs=4) for t in targets]
        rep = write_report(outdir, results, probes)
        rows = {r["slug"]: r for r in rep["rows"]}
        g, b = rows["mock-good"], rows["mock-degraded"]
        print(f"    mock-good     visual={g['visual']} capability={g['capability']} "
              f"precision={g['precision']} agree={g['top1_agreement']}")
        print(f"    mock-degraded visual={b['visual']} capability={b['capability']} "
              f"precision={b['precision']} agree={b['top1_agreement']}")
        checks = [
            ("正常端点的视觉分 > 90%", (g["visual"] or 0) > 0.9),
            ("被量化端点的视觉分明显更低", (g["visual"] or 0) - (b["visual"] or 0) > 0.2),
            ("被量化端点的能力分明显更低", (g["capability"] or 0) - (b["capability"] or 0) > 0.15),
            ("正常端点识别为 bf16", g["precision"] == "bf16"),
            ("被量化端点识别为更低精度", _MANTISSA_BITS.get(b["precision"], 99) < _MANTISSA_BITS.get(g["precision"], 0)),
            ("被量化端点被标出可疑", any(f["sev"] in ("high", "warn") for f in b["flags"])),
            ("生成了 gallery.html", os.path.exists(os.path.join(outdir, "gallery.html"))),
        ]
        for label, hit in checks:
            ok &= hit
            print(f"    {'✓' if hit else '✗'} {label}")
        print(f"\n    报告：{rep['md']}")
        print(f"    图表：{os.path.join(outdir, 'gallery.html')}")
    finally:
        stop_g(); stop_b()

    print("\n" + "=" * 68)
    print("自检结果：" + ("全部通过 ✅" if ok else "存在失败 ❌"))
    print("=" * 68)
    return 0 if ok else 1


def _print_summary(rows: list):
    """打印总览表 + 所有标记。在线/离线两种模式共用。"""
    print()
    print(f"{'模型':<24} {'L1视觉':>7} {'L2能力':>7} {'logit精度':>12} {'答对率':>7}  判定")
    print("-" * 78)
    for r in sorted(rows, key=lambda x: -(x["visual"] or 0)):
        prec = "离线/不支持" if (r.get("offline") or not r["precision"]) else r["precision"]
        print(f"{r['name'][:24]:<24} {_fmt_pct(r['visual']):>7} {_fmt_pct(r['capability']):>7} "
              f"{prec:>12} {_fmt_pct(r['top1_agreement']):>7}  {verdict_line(r)}")
    print()
    for r in rows:
        for fl in r["flags"]:
            mark = {"high": "⛔", "warn": "⚠️", "info": "ℹ️"}[fl["sev"]]
            print(f"{mark} {r['name']}：{fl['msg']}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(
        prog="modelcheck",
        description="模型成色体检：视觉探针 + 确定性探针 + logprob/量化指纹",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--targets", help="端点配置文件（见 targets.example.json）")
    ap.add_argument("--out", default="runs/latest", help="输出目录")
    ap.add_argument("--only", default="", help="只跑某些探针/分组，逗号分隔（visual,capability 或具体 id）")
    ap.add_argument("--jobs", type=int, default=1, help="并发数（同一端点内并行跑探针）")
    ap.add_argument("--no-exec", action="store_true", help="禁止执行模型生成的代码（默认会跑）")
    ap.add_argument("--no-fingerprint", action="store_true", help="跳过 L3 指纹（省请求）")
    ap.add_argument("--baseline", help="已有基线 JSON，用于检测端点漂移")
    ap.add_argument("--save-baseline", help="把本次关键指标存成基线 JSON")
    ap.add_argument("--selftest", action="store_true", help="不需要 API key 的自检")
    ap.add_argument("--list-probes", action="store_true", help="列出所有探针")
    ap.add_argument("--emit-prompts", metavar="DIR",
                    help="离线模式：把题目导出成可粘贴清单（不调用任何 API）")
    ap.add_argument("--import-dir", metavar="DIR",
                    help="离线模式：从人工收集的回答目录生成报告（不联网）")
    args = ap.parse_args(argv)

    if args.selftest:
        return selftest()

    probes = build_probes()
    if args.list_probes:
        for p in probes:
            print(f"{p.group:12s} {p.id:20s} {p.desc}")
        print(f"{'fingerprint':12s} {'logprob_battery':20s} 量化检测：分布糊度 + logit 精度反推")
        print(f"{'fingerprint':12s} {'tokenizer_fp':20s} 换模型检测：固定字符串的 prompt_tokens")
        print(f"{'fingerprint':12s} {'protocol':20s} system 遵循 / model 回显 / 温度 0 确定性")
        return 0

    if args.only:
        want = {x.strip() for x in args.only.split(",") if x.strip()}
        probes = [p for p in probes if p.id in want or p.group in want]
        if not probes:
            ap.error(f"--only={args.only} 没有匹配到任何探针")

    def log(msg):
        print(msg, flush=True)

    # ---- 离线模式 1：只导出题目 ----
    if args.emit_prompts:
        md = emit_prompts(args.emit_prompts, probes, log=log)
        print(f"\n清单：{md}")
        print(f"题目：{os.path.join(args.emit_prompts, 'prompts')}/")
        print("\n按 PROMPTS.md 的约定收集回答后，再跑：")
        print(f"  python3 {os.path.basename(__file__)} --import-dir answers/ --out {args.out}")
        return 0

    # ---- 离线模式 2：从人工收集的回答出报告 ----
    if args.import_dir:
        log(f"modelcheck v{__version__}　离线模式（人工收集）　探针 {len(probes)} 个")
        results = load_manual_answers(args.import_dir, probes,
                                      exec_code=not args.no_exec, log=log)
        if not results:
            log("没有可用的作答")
            return 1
        rep = write_report(args.out, results, probes)
        _print_summary(rep["rows"])
        if args.save_baseline:
            with open(args.save_baseline, "w", encoding="utf-8") as f:
                json.dump(baseline_snapshot(results), f, ensure_ascii=False, indent=2)
            print(f"\n基线已保存：{args.save_baseline}")
        print(f"\n报告：{rep['md']}")
        print(f"图表：{os.path.join(args.out, 'gallery.html')}")
        return 0

    if not args.targets:
        ap.error("需要 --targets，或用 --emit-prompts / --import-dir / --selftest / --list-probes")
    targets = load_targets(args.targets)

    log(f"modelcheck v{__version__}　端点 {len(targets)} 个　探针 {len(probes)} 个")
    results = []
    for t in targets:
        try:
            results.append(run_target(t, probes, args.out, exec_code=not args.no_exec,
                                      jobs=max(1, args.jobs),
                                      do_fingerprint=not args.no_fingerprint, log=log))
        except KeyboardInterrupt:
            raise
        except Exception as e:
            log(f"  [{t.name}] 失败：{type(e).__name__}: {e}")

    if not results:
        log("没有任何端点成功完成")
        return 1

    rep = write_report(args.out, results, probes)
    rows = rep["rows"]
    _print_summary(rows)

    if args.baseline:
        with open(args.baseline, encoding="utf-8") as f:
            try:
                base = json.load(f)
            except Exception:
                base = {}
        print("\n漂移对比（vs 基线）：")
        for fl in compare_baseline(results, base):
            mark = {"high": "⛔", "warn": "⚠️", "info": "ℹ️"}[fl["sev"]]
            print(f"{mark} {fl['slug']}：{fl['msg']}")

    if args.save_baseline:
        with open(args.save_baseline, "w", encoding="utf-8") as f:
            json.dump(baseline_snapshot(results), f, ensure_ascii=False, indent=2)
        print(f"\n基线已保存：{args.save_baseline}")

    print(f"\n报告：{rep['md']}")
    print(f"图表：{os.path.join(args.out, 'gallery.html')}")
    print(f"原始：{os.path.join(args.out, 'report.json')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
