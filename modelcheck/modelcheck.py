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
import json
import math
import os
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
        s = re.sub(r"[^A-Za-z0-9._-]+", "-", self.name).strip("-")
        return s or "target"

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
                if s.w < 1.3 * a.body.w:
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
        _mk(crit, "valid_svg", "SVG 合法性", 0.0, 0.10, a.reason)
    elif not (a.has_viewbox or a.has_width_height):
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

    def search(fmt, lo, hi, step):
        c = lo
        while c <= hi:
            if all(_representable(_f32(v + c), fmt) for v in vals):
                return c
            c += step
        return None

    hits = {}
    for fmt in _FORMATS:
        step = 1 / 256
        c = search(fmt, -8.0, 40.0, step)
        if c is not None:
            c2 = search(fmt, c - step, c + step, step / 256)
            hits[fmt] = round(c2 if c2 is not None else c, 6)

    if not hits:
        return {"ok": False, "reason": "没有找到任何标准浮点格式能解释这批 logprob"}
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
    try:
        obj = json.loads(body)
    except Exception as e:
        return 0.0, f"不是合法 JSON：{e}"
    if not isinstance(obj, dict):
        return 0.2, f"顶层不是对象，而是 {type(obj).__name__}"
    score, notes = 1.0, []
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
            kind="svg", desc="换一个同样罕见的组合，排除背答案"))
    A(Probe("scene_control", "visual", "Generate an SVG of a house with a door and two windows",
            kind="svg", max_tokens=2000,
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
    m = re.search(r"<svg[\s\S]*?</svg>", text, re.I)
    if m:
        return m.group(0).strip()
    m = re.search(r"<!DOCTYPE html[\s\S]*", text, re.I)
    if m:
        return m.group(0).strip()
    m = re.search(r"<html[\s\S]*?</html>", text, re.I)
    if m:
        return m.group(0).strip()
    return text.strip()

SVG_NS = "http://www.w3.org/2000/svg"
