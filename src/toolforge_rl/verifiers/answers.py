"""Deterministic numeric, symbolic-math, and extractive-QA verification."""

from __future__ import annotations

import math
import re
import string
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction


@dataclass(frozen=True)
class VerificationResult:
    correct: bool
    normalized_prediction: str
    normalized_reference: str
    reason: str


_BOX_RE = re.compile(r"\\boxed\s*\{([^{}]*(?:\{[^{}]*\}[^{}]*)*)\}")
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.I)


def _strip_answer_wrapper(text: object) -> str:
    value = str(text).strip()
    boxes = _BOX_RE.findall(value)
    if boxes:
        value = boxes[-1]
    answer_tags = re.findall(r"<answer>\s*(.*?)\s*</answer>", value, re.I | re.S)
    if answer_tags:
        value = answer_tags[-1]
    return value.strip()


def normalize_numeric(value: object) -> str:
    text = _strip_answer_wrapper(value)
    text = text.replace(",", "").replace("−", "-").replace("%", " percent")
    text = re.sub(r"^[\s$€£¥]+", "", text)
    match = re.search(r"[-+]?\d+(?:\.\d+)?(?:\s*/\s*[-+]?\d+(?:\.\d+)?)?", text)
    if not match:
        return ""
    token = re.sub(r"\s+", "", match.group(0))
    try:
        if "/" in token:
            return str(Fraction(token))
        decimal = Decimal(token)
        if decimal == decimal.to_integral():
            return str(decimal.quantize(Decimal(1)))
        return format(decimal.normalize(), "f")
    except (InvalidOperation, ValueError, ZeroDivisionError):
        return ""


def _as_float(normalized: str) -> float | None:
    try:
        return float(Fraction(normalized)) if "/" in normalized else float(normalized)
    except (ValueError, ZeroDivisionError):
        return None


def verify_numeric(
    prediction: object,
    reference: object,
    *,
    rel_tol: float = 1e-6,
    abs_tol: float = 1e-6,
) -> VerificationResult:
    pred = normalize_numeric(prediction)
    ref = normalize_numeric(reference)
    pred_value, ref_value = _as_float(pred), _as_float(ref)
    if pred_value is None or ref_value is None:
        return VerificationResult(False, pred, ref, "numeric_parse_failure")
    correct = math.isclose(pred_value, ref_value, rel_tol=rel_tol, abs_tol=abs_tol)
    return VerificationResult(correct, pred, ref, "numeric_match" if correct else "numeric_mismatch")


def normalize_qa(value: object) -> str:
    text = _strip_answer_wrapper(value).lower().replace("_", " ")
    text = "".join(" " if char in string.punctuation else char for char in text)
    text = _ARTICLES_RE.sub(" ", text)
    return " ".join(text.split())


def verify_qa(
    prediction: object, reference: object, *, aliases: list[str] | tuple[str, ...] = ()
) -> VerificationResult:
    pred = normalize_qa(prediction)
    refs = [normalize_qa(reference), *(normalize_qa(alias) for alias in aliases)]
    correct = bool(pred) and pred in refs
    return VerificationResult(correct, pred, refs[0], "qa_exact_or_alias" if correct else "qa_mismatch")


def verify_math(prediction: object, reference: object) -> VerificationResult:
    pred_text = _strip_answer_wrapper(prediction)
    ref_text = _strip_answer_wrapper(reference)
    try:
        from math_verify import parse, verify

        def parsed(value: str):
            result = parse(value)
            if not result and not (value.startswith("$") and value.endswith("$")):
                result = parse(f"${value}$")
            return result

        correct = bool(verify(parsed(ref_text), parsed(pred_text)))
        return VerificationResult(correct, pred_text, ref_text, "math_verify" if correct else "math_mismatch")
    except Exception as exc:  # library parser is intentionally best-effort
        numeric = verify_numeric(pred_text, ref_text)
        if numeric.normalized_prediction and numeric.normalized_reference:
            return VerificationResult(
                numeric.correct,
                numeric.normalized_prediction,
                numeric.normalized_reference,
                "math_numeric_fallback" if numeric.correct else "math_mismatch",
            )
        return VerificationResult(False, pred_text, ref_text, f"math_parse_failure:{type(exc).__name__}")


def verify_answer(
    verifier_type: str,
    prediction: object,
    reference: object,
    *,
    aliases: list[str] | tuple[str, ...] = (),
) -> VerificationResult:
    kind = verifier_type.lower()
    if kind in {"gsm8k", "numeric", "retrieve_then_compute"}:
        return verify_numeric(prediction, reference)
    if kind in {"math", "math500"}:
        return verify_math(prediction, reference)
    if kind in {"hotpotqa", "2wiki", "qa", "exact"}:
        return verify_qa(prediction, reference, aliases=aliases)
    raise ValueError(f"unknown verifier_type: {verifier_type}")
