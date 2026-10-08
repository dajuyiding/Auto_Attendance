#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""modelcheck 的回归测试。

    python3 -m unittest test_modelcheck -v
    python3 test_modelcheck.py               # 等价
"""

import io
import math
import os
import random
import struct
import unittest

import modelcheck as M

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


def read_fixture(name):
    with io.open(os.path.join(FIXTURES, name), encoding="utf-8") as f:
        return f.read()


def _snap(v, fmt):
    if fmt == "fp32":
        return M._f32(v)
    if fmt == "fp16":
        try:
            return struct.unpack("<e", struct.pack("<e", v))[0]
        except OverflowError:
            return v
    return M._nearest(v, fmt)


def synth_logprobs(fmt, seed=1, n=20, lo=-5.0, hi=25.0):
    """造一批「已知 logits 精度」的 logprob，模拟真实 API 返回。"""
    rnd = random.Random(seed)
    logits = [_snap(rnd.uniform(lo, hi), fmt) for _ in range(n)]
    mx = max(logits)
    lse = mx + math.log(sum(math.exp(v - mx) for v in logits))
    return sorted({M._f32(v - lse) for v in logits}, reverse=True)


HOUSE_SVG = ('<svg width="400" height="300" viewBox="0 0 400 300" '
             'xmlns="http://www.w3.org/2000/svg">'
             '<rect x="120" y="140" width="160" height="120" fill="#e8dcc8" '
             'stroke="#8a7a66" stroke-width="4"/>'
             '<polygon points="120,140 200,80 280,140" fill="#b04a3a"/>'
             '<rect x="178" y="192" width="48" height="68" fill="#7a5230"/>'
             '<circle cx="214" cy="228" r="4" fill="#f0d060"/>'
             '<rect x="138" y="164" width="36" height="36" fill="#a8d8f0" '
             'stroke="#8a7a66" stroke-width="3"/>'
             '<rect x="226" y="164" width="36" height="36" fill="#a8d8f0" '
             'stroke="#8a7a66" stroke-width="3"/>'
             '<line x1="156" y1="164" x2="156" y2="200" stroke="#8a7a66" stroke-width="3"/>'
             '<line x1="138" y1="182" x2="174" y2="182" stroke="#8a7a66" stroke-width="3"/>'
             '</svg>')


class TestSvgScoring(unittest.TestCase):
    def test_fixtures_exist(self):
        self.assertTrue(os.path.isdir(FIXTURES), "fixtures/ 目录缺失")
        self.assertTrue(len(os.listdir(FIXTURES)) >= 4)

    def test_good_beats_bad(self):
        good = M.score_svg(
            M.analyze_svg(read_fixture("claude-3-5-sonnet-20241022.svg")))["score"]
        bad = M.score_svg(
            M.analyze_svg(read_fixture("gpt-3.5-turbo.svg")))["score"]
        self.assertGreater(good - bad, 0.15, f"好样例 {good} 没有明显高于差样例 {bad}")

    def test_claude_pelican_detects_all_parts(self):
        a = M.analyze_svg(read_fixture("claude-3-5-sonnet-20241022.svg"))
        self.assertIsNotNone(a.wheel_pair, "没有识别出两个车轮")
        self.assertIsNotNone(a.body, "没有识别出身体")
        self.assertIsNotNone(a.head, "没有识别出头部")
        self.assertIsNotNone(a.beak, "没有识别出喙")
        self.assertGreaterEqual(a.n_legs, 2, "腿的数量不足")

    def test_head_is_not_the_eye(self):
        """头必须是「有份量」的块，不能被眼睛那种小圆点顶替。"""
        a = M.analyze_svg(read_fixture("claude-3-5-sonnet-20241022.svg"))
        self.assertGreater(a.head.w, 0.15 * a.body.w)

    def test_empty_and_garbage(self):
        self.assertFalse(M.analyze_svg("").ok)
        self.assertFalse(M.analyze_svg("这里没有任何 SVG").ok)
        self.assertEqual(M.score_svg(M.analyze_svg(""))["score"], 0.0)

    def test_generic_scorer_ignores_pelican_requirements(self):
        """通用打分器不应该要求「两个轮子」，房子也要能得高分。"""
        self.assertGreater(M.score_svg_generic(M.analyze_svg(HOUSE_SVG))["score"], 0.7)
        # 而鹈鹕专用打分器必须认为它缺件（因为不是自行车）
        self.assertLess(M.score_svg(M.analyze_svg(HOUSE_SVG))["score"], 0.5)

    def test_cleanliness_penalises_text_only(self):
        lazy = ('<svg viewBox="0 0 100 100">'
                '<text x="5" y="50">a pelican on a bicycle</text></svg>')
        self.assertLess(M.score_svg(M.analyze_svg(lazy))["score"], 0.35)



class TestPrecisionInference(unittest.TestCase):
    def test_identifies_all_five_formats(self):
        for fmt in ("fp8_e5m2", "fp8_e4m3", "bf16", "fp16", "fp32"):
            for seed in (1, 2):
                got = M.infer_logit_precision(synth_logprobs(fmt, seed))["precision"]
                self.assertEqual(got, fmt, f"{fmt} seed={seed} 被判成 {got}")

    def test_no_false_positive_on_random_fp32(self):
        """完全随机的 fp32 分布不能被误判成低精度（否则会把正常模型标成量化）。"""
        for seed in range(20):
            got = M.infer_logit_precision(synth_logprobs("fp32", 500 + seed))["precision"]
            self.assertEqual(got, "fp32", f"seed={seed} 误报为 {got}")

    def test_too_few_values(self):
        self.assertFalse(M.infer_logit_precision([-0.1, -0.2])["ok"])

    def test_mantissa_ordering(self):
        self.assertLess(M._MANTISSA_BITS["fp8_e5m2"], M._MANTISSA_BITS["fp8_e4m3"])
        self.assertLess(M._MANTISSA_BITS["fp8_e4m3"], M._MANTISSA_BITS["bf16"])
        self.assertLess(M._MANTISSA_BITS["bf16"], M._MANTISSA_BITS["fp16"])
        self.assertLess(M._MANTISSA_BITS["fp16"], M._MANTISSA_BITS["fp32"])

    def test_nesting_is_expected(self):
        """粗格式能被细数据匹配，这是已知的「精度碰撞」，all_matching 要如实反映。"""
        r = M.infer_logit_precision(synth_logprobs("bf16", 1))
        self.assertIn("bf16", r["all_matching"])
        self.assertIn("fp16", r["all_matching"])


class TestLogprobStats(unittest.TestCase):
    @staticmethod
    def _lp(pairs):
        return [{"token": "x", "logprob": pairs[0],
                 "top_logprobs": [{"token": f"t{i}", "logprob": v}
                                  for i, v in enumerate(pairs)]}]

    def test_confident_distribution(self):
        st = M.logprob_stats(self._lp([-0.01, -8.0, -12.0]))
        self.assertGreater(st["mean_top1_prob"], 0.95)
        self.assertLess(st["mean_entropy"], 0.3)

    def test_flat_distribution(self):
        st = M.logprob_stats(self._lp([-1.4, -1.4, -1.4, -1.4, -1.4]))
        self.assertLess(st["mean_top1_prob"], 0.3)
        self.assertGreater(st["mean_entropy"], 1.0)

    def test_missing(self):
        self.assertFalse(M.logprob_stats([])["ok"])

    def test_position_logprobs(self):
        seq = self._lp([-0.01, -8.0, -12.0])
        self.assertEqual(len(M.position_logprobs(seq, 0)), 3)
        self.assertEqual(M.position_logprobs(seq, 5), [])


class TestExtractCode(unittest.TestCase):
    def test_fenced_svg(self):
        t = "好的，这是图：\n```svg\n<svg xmlns='x'><circle r='1'/></svg>\n```\n希望有帮助"
        out = M.extract_code(t)
        self.assertTrue(out.startswith("<svg"))
        self.assertNotIn("希望有帮助", out)

    def test_fenced_python(self):
        t = "```python\ndef f():\n    return 1\n```"
        self.assertIn("def f", M.extract_code(t, lang_hints=("python",)))

    def test_plain_html_wins_over_inner_svg(self):
        """完整 HTML 里含 SVG 时不能只抠出 SVG，否则动画/CSS 会被丢掉。"""
        t = ("<!DOCTYPE html>\n<html><head><style>"
             "@keyframes s{to{transform:rotate(360deg)}}"
             "</style></head><body><svg viewBox='0 0 10 10'>"
             "<circle r='1'/></svg></body></html>")
        out = M.extract_code(t)
        self.assertIn("<!DOCTYPE html", out)
        self.assertIn("@keyframes", out)

    def test_plain_svg(self):
        t = "这是结果\n<svg viewBox='0 0 10 10'><rect width='1' height='1'/></svg>\n完"
        out = M.extract_code(t)
        self.assertTrue(out.startswith("<svg"))
        self.assertTrue(out.endswith("</svg>"))




class TestCheckers(unittest.TestCase):
    @staticmethod
    def _r(text, finish="stop"):
        return M.CallResult(ok=True, text=text, finish_reason=finish)

    def test_exact(self):
        s, _ = M.check_exact("3309836", {"expect": "3309836"}, self._r("3309836"))
        self.assertEqual(s, 1.0)
        s, _ = M.check_exact("3310636", {"expect": "3309836"}, self._r("3310636"))
        self.assertEqual(s, 0.0)
        # 答案对，但絮絮叨叨说了半天 -> 扣分
        s, _ = M.check_exact("First 4783 * 692 = 3309836. So the answer is 3309836.",
                             {"expect": "3309836"}, self._r(""))
        self.assertLess(s, 1.0)

    def test_letters(self):
        s, _ = M.check_letters("3, 3, 4", {"expect": [3, 3, 4]}, self._r(""))
        self.assertEqual(s, 1.0)
        s, _ = M.check_letters("2 3 4", {"expect": [3, 3, 4]}, self._r(""))
        self.assertEqual(s, 0.0)

    def test_json_strict(self):
        params = {"schema": {"name": "str", "age": "int", "tags": "array"}, "array_len": 3}
        s, _ = M.check_json('{"name":"a","age":1,"tags":["x","y","z"]}', params, self._r(""))
        self.assertEqual(s, 1.0)
        s, _ = M.check_json('```json\n{"name":"a","age":1,"tags":["x","y","z"]}\n```',
                            params, self._r(""))
        self.assertLess(s, 1.0, "套了 markdown 围栏应当扣分")
        s, _ = M.check_json('{"name":"a","age":"old","tags":["x","y","z"]}',
                            params, self._r(""))
        self.assertLess(s, 1.0, "age 类型错了应当扣分")
        s, _ = M.check_json('不是 JSON', params, self._r(""))
        self.assertEqual(s, 0.0)

    def test_words(self):
        params = {"n_words": 5, "last_word": "banana",
                  "lowercase": True, "no_punct": True}
        s, _ = M.check_words("please pass the ripe banana", params, self._r(""))
        self.assertEqual(s, 1.0)
        s, _ = M.check_words("I like yellow banana", params, self._r(""))
        self.assertLess(s, 1.0)

    def test_sequence(self):
        full = "\n".join(str(i) for i in range(1, 201))
        s, _ = M.check_sequence(full, {"n": 200}, self._r(full))
        self.assertEqual(s, 1.0)
        part = "\n".join(str(i) for i in range(1, 121))
        s, _ = M.check_sequence(part, {"n": 200}, self._r(part))
        self.assertLessEqual(s, 0.6)
        looped = "\n".join(["1"] * 700)
        s, _ = M.check_sequence(looped, {"n": 200}, self._r(looped))
        self.assertLess(s, 0.3)

    def test_chinese(self):
        good = ("量化是指把模型参数用更低位数的整数存储，从而在牺牲一定精度的前提下，"
                "显著降低显存占用并提升推理速度。常见的做法包括 INT8 与 INT4，"
                "幅度过大的量化会让模型在数值推理和长文本任务上明显退化。")
        s, _ = M.check_chinese(good, {"min_han": 60, "must_contain": ["精度", "显存"]},
                               self._r(""))
        self.assertEqual(s, 1.0)
        s, _ = M.check_chinese("量化就是压缩。", {"min_han": 60}, self._r(""))
        self.assertLess(s, 0.6)
        s, _ = M.check_chinese("啊" * 200, {"min_han": 60}, self._r(""))
        self.assertLess(s, 1.0, "复读应当扣分")

    def test_needle(self):
        s, _ = M.check_needle("编号是 XZ-4471", {"needle": "XZ-4471"}, self._r(""))
        self.assertEqual(s, 1.0)
        s, _ = M.check_needle("我不知道", {"needle": "XZ-4471"}, self._r(""))
        self.assertEqual(s, 0.0)

    def test_python_executes(self):
        good = ("def median(nums):\n    s = sorted(nums)\n    n = len(s)\n"
                "    if n % 2:\n        return s[n // 2]\n"
                "    return (s[n // 2 - 1] + s[n // 2]) / 2\n")
        tests = ("assert median([1,2,3,4]) == 2.5\n"
                 "assert median([3,1,2]) == 2\nprint('ok')\n")
        s, det = M.check_python(good, {"tests": tests}, self._r(good))
        self.assertEqual(s, 1.0, det)
        bad = "def median(nums):\n    s = sorted(nums)\n    return s[len(s) // 2]\n"
        s, det = M.check_python(bad, {"tests": tests}, self._r(bad))
        self.assertEqual(s, 0.0)

    def test_python_timeout_is_handled(self):
        code = "def median(nums):\n    while True:\n        pass\n"
        s, det = M.check_python(code, {"tests": "median([1])\n", "timeout": 2},
                                self._r(code))
        self.assertEqual(s, 0.0)
        self.assertIn("超时", det)


class TestProbes(unittest.TestCase):
    def test_build_probes_unique_ids(self):
        ps = M.build_probes()
        ids = [p.id for p in ps]
        self.assertEqual(len(ids), len(set(ids)), "探针 id 有重复")
        self.assertTrue(all(p.group in ("visual", "capability") for p in ps))

    def test_all_checkers_registered(self):
        for p in M.build_probes():
            if p.checker:
                self.assertIn(p.checker, M.CHECKERS, f"{p.id} 用了未注册的检查器")

    def test_needle_probe_actually_contains_needle(self):
        p = [x for x in M.build_probes() if x.id == "needle"][0]
        self.assertIn("XZ-4471", p.prompt)
        self.assertGreater(len(p.prompt), 3000)


class TestReportAndVerdict(unittest.TestCase):
    @staticmethod
    def _fake_result(slug, model, visual, capability, precision, agreement=1.0,
                     tokenizer=None):
        runs = [
            {"probe_id": "pelican_svg", "group": "visual", "kind": "svg", "desc": "",
             "weight": 1.0, "ok": True, "score": visual, "detail": "", "error": "",
             "text": "", "code": "", "criteria": [], "analysis": {}, "latency_ms": 0,
             "finish_reason": "stop", "consistency": None, "scored": True, "extra": {}},
            # 没有检查器的探针：scored=False，绝不能计入平均分
            {"probe_id": "selfid", "group": "capability", "kind": "text", "desc": "",
             "weight": 0.5, "ok": True, "score": 0.0, "detail": "x", "error": "",
             "text": "", "code": "", "criteria": [], "analysis": {}, "latency_ms": 0,
             "finish_reason": "stop", "consistency": None, "scored": False, "extra": {}},
        ]
        if capability is not None:
            runs.append({"probe_id": "arith_mul", "group": "capability", "kind": "text",
                         "desc": "", "weight": 1.0, "ok": True, "score": capability,
                         "detail": "", "error": "", "text": "", "code": "", "criteria": [],
                         "analysis": {}, "latency_ms": 0, "finish_reason": "stop",
                         "consistency": None, "scored": True, "extra": {}})
        return {
            "target": {"name": slug, "model": model, "base_url": "http://x", "api_key": "",
                       "note": "", "extra_body": {}, "headers": {}},
            "slug": slug, "runs": runs,
            "fingerprint": {
                "logprob": {"ok": True,
                            "precision": {"ok": True, "precision": precision,
                                          "mantissa_bits": M._MANTISSA_BITS[precision],
                                          "all_matching": [precision]},
                            "top1_agreement": agreement, "mean_top1_prob": 0.9,
                            "mean_entropy": 0.2, "per_prompt": []},
                "tokenizer": {"ok": True, "fingerprint": tokenizer, "counts": {}},
                "protocol": {"deterministic": True, "model_echo": model},
            },
        }

    def test_scoreless_probes_excluded_from_average(self):
        res = self._fake_result("a", "m", 0.8, 1.0, "bf16")
        gs = M.group_scores(res["runs"])
        self.assertAlmostEqual(gs["capability"], 1.0, places=6)  # selfid 的 0 分不计入

    def test_bf16_is_not_flagged_as_suspicious(self):
        """BF16 是现代推理常态（GPT-4o 就是），不能被标成可疑。"""
        rows = M.analyze_fleet([self._fake_result("a", "m1", 0.9, 0.9, "bf16"),
                                self._fake_result("b", "m2", 0.9, 0.9, "bf16")])
        for r in rows:
            bad = [f for f in r["flags"]
                   if f["sev"] in ("high", "warn") and f["kind"] == "low_precision"]
            self.assertFalse(bad, f"BF16 被误标：{r['flags']}")
            self.assertEqual(M.verdict_line(r), "✅ 正常")

    def test_fp8_is_flagged(self):
        rows = M.analyze_fleet([
            self._fake_result("a", "m1", 0.9, 0.9, "bf16"),
            self._fake_result("b", "m2", 0.4, 0.3, "fp8_e4m3", agreement=0.2)])
        bad = [r for r in rows if r["slug"] == "b"][0]
        self.assertIn("low_precision", {f["kind"] for f in bad["flags"]})
        self.assertEqual(M.verdict_line(bad), "⛔ 可疑")

    def test_tokenizer_mismatch_detected(self):
        rows = M.analyze_fleet([
            self._fake_result("a", "gpt-4o", 0.9, 0.9, "bf16", tokenizer=[2, 11, 1, 5, 2]),
            self._fake_result("b", "gpt-4o", 0.9, 0.9, "bf16", tokenizer=[3, 12, 1, 6, 2])])
        for r in rows:
            self.assertIn("tokenizer_mismatch", {f["kind"] for f in r["flags"]})

    def test_same_tokenizer_no_flag(self):
        rows = M.analyze_fleet([
            self._fake_result("a", "gpt-4o", 0.9, 0.9, "bf16", tokenizer=[2, 11, 1, 5, 2]),
            self._fake_result("b", "gpt-4o", 0.9, 0.9, "bf16", tokenizer=[2, 11, 1, 5, 2])])
        for r in rows:
            self.assertNotIn("tokenizer_mismatch", {f["kind"] for f in r["flags"]})

    def test_baseline_drift(self):
        res = self._fake_result("a", "m", 0.9, 0.9, "bf16")
        res["fingerprint"]["logprob"]["per_prompt"] = [
            {"prompt": "2 + 2 =", "top1_prob": 0.99, "expected": "4",
             "got": "4", "match": True}]
        base = M.baseline_snapshot([res])
        res2 = self._fake_result("a", "m", 0.9, 0.9, "bf16")
        res2["fingerprint"]["logprob"]["per_prompt"] = [
            {"prompt": "2 + 2 =", "top1_prob": 0.60, "expected": "4",
             "got": "4", "match": True}]
        flags = M.compare_baseline([res2], base, tol=0.05)
        self.assertTrue(any(f["sev"] in ("high", "warn") for f in flags), flags)

    def test_baseline_no_drift(self):
        res = self._fake_result("a", "m", 0.9, 0.9, "bf16")
        res["fingerprint"]["logprob"]["per_prompt"] = [
            {"prompt": "2 + 2 =", "top1_prob": 0.99, "expected": "4",
             "got": "4", "match": True}]
        base = M.baseline_snapshot([res])
        flags = M.compare_baseline([res], base, tol=0.05)
        self.assertFalse(any(f["sev"] in ("high", "warn") for f in flags), flags)



class TestEndToEnd(unittest.TestCase):
    def test_mock_pipeline_flags_degraded(self):
        import tempfile
        url_good, stop_g = M.start_mock_gateway("good", "bf16")
        url_bad, stop_b = M.start_mock_gateway("degraded", "fp8_e4m3")
        try:
            probes = [p for p in M.build_probes()
                      if p.group == "visual"
                      or p.id in ("arith_mul", "count_letters", "json_strict", "code_fix")]
            targets = [M.Target("good", url_good, "k", "mock-1"),
                       M.Target("bad", url_bad, "k", "mock-2")]
            with tempfile.TemporaryDirectory() as td:
                results = [M.run_target(t, probes, td, jobs=4) for t in targets]
                rep = M.write_report(td, results, probes)
                self.assertTrue(os.path.exists(os.path.join(td, "gallery.html")))
                self.assertTrue(os.path.exists(os.path.join(td, "report.md")))
                self.assertTrue(os.path.exists(os.path.join(td, "report.json")))
                rows = {r["slug"]: r for r in rep["rows"]}
                self.assertGreater(rows["good"]["visual"], 0.85)
                self.assertGreater(rows["good"]["visual"] - rows["bad"]["visual"], 0.2)
                self.assertGreater(rows["good"]["capability"]
                                   - rows["bad"]["capability"], 0.15)
                self.assertEqual(rows["good"]["precision"], "bf16")
                self.assertLess(M._MANTISSA_BITS[rows["bad"]["precision"]],
                                M._MANTISSA_BITS[rows["good"]["precision"]])
                self.assertEqual(M.verdict_line(rows["bad"]), "⛔ 可疑")
        finally:
            stop_g()
            stop_b()


if __name__ == "__main__":
    unittest.main(verbosity=2)

