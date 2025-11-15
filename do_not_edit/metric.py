"""do-not-edit metric shim that mirrors core metric implementation.

This file intentionally mirrors `/home/25-InThon/src/core/metric.py` so that
example code that imports `do_not_edit.metric.compute_metrics` gets identical
behavior to the central `src/core/metric.py` implementation.
"""

from typing import List, Tuple, Dict, Optional
import math

def normalize_pred(s: str) -> str:
    """정규화: 앞쪽 0 제거, 빈 문자열은 '0'으로 처리한다."""
    if s is None:
        return ''
    # 안전을 위해 문자열로 변환
    try:
        s = str(s)
    except Exception:
        return ''
    # strip leading zeros
    r = s.lstrip('0')
    if r == '':
        r = '0'

    # strip trailing fractional zeros: 123.00 -> 123 ; 123.4500 -> 123.45 ; 0.0 -> 0
    if '.' in r:
        # remove trailing zeros after decimal
        r = r.rstrip('0')
        # if decimal point is last, remove it
        if r.endswith('.'):
            r = r[:-1]
        if r == '':
            r = '0'

    return r


def exact_match(pred: str, gold: str) -> int:
    """정규화 후 정확히 같으면 1, 아니면 0 반환한다."""
    return 1 if normalize_pred(pred) == normalize_pred(gold) else 0


def token_edit_similarity(pred: str, gold: str) -> float:
    """Token Edit Similarity (TES).

    TES = 1 - LevenshteinDistance(pred, gold) / max(len(pred), len(gold))
    빈 문자열 등 분모가 0인 경우는 0으로 처리한다.
    """
    a = normalize_pred(pred)
    b = normalize_pred(gold)
    la = len(a)
    lb = len(b)
    if max(la, lb) == 0:
        return 1.0

    # simple Levenshtein distance
    dp = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        dp[i][0] = i
    for j in range(lb + 1):
        dp[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)
    dist = dp[la][lb]
    tes = 1.0 - (dist / max(la, lb))
    return max(0.0, tes)


def batch_metrics(preds: List[str], golds: List[str]) -> dict:
    """배치 단위로 EM, TES 평균을 계산하여 반환한다."""
    assert len(preds) == len(golds), "preds와 golds 길이가 달라서 계산할 수 없다."
    n = len(preds)
    em_sum = 0
    tes_sum = 0.0
    for p, g in zip(preds, golds):
        em_sum += exact_match(p, g)
        tes_sum += token_edit_similarity(p, g)
    return {
        'EM': em_sum / n if n > 0 else 0.0,
        'TES': tes_sum / n if n > 0 else 0.0,
    }


def _group_indices(group_ids: Optional[List[Optional[str]]]) -> Dict[str, List[int]]:
    groups: Dict[str, List[int]] = {}
    if group_ids is None:
        return groups
    for i, gid in enumerate(group_ids):
        if gid is None:
            continue
        groups.setdefault(str(gid), []).append(i)
    return groups


def equational_consistency(preds: List[str], refs: List[str], group_ids: Optional[List[Optional[str]]]) -> float:
    groups = _group_indices(group_ids)
    eligible = [idxs for idxs in groups.values() if len(idxs) == 2]
    if not eligible:
        return 0.0
    ok = 0
    for idxs in eligible:
        p = [normalize_pred(preds[i]) for i in idxs]
        r = [normalize_pred(refs[i]) for i in idxs]
        if p[0] == p[1] and p[0] == r[0] == r[1]:
            ok += 1
    return ok / len(eligible)


def reasoning_consistency(preds: List[str], refs: List[str], group_ids: Optional[List[Optional[str]]]) -> float:
    groups = _group_indices(group_ids)
    eligible = [idxs for idxs in groups.values() if len(idxs) >= 3]
    if not eligible:
        return 0.0
    ok = 0
    for idxs in eligible:
        pset = {normalize_pred(preds[i]) for i in idxs}
        rset = {normalize_pred(refs[i]) for i in idxs}
        if len(pset) == 1 and len(rset) == 1 and next(iter(pset)) == next(iter(rset)):
            ok += 1
    return ok / len(eligible)


def compute_metrics(preds: List[str], refs: List[str], group_ids: Optional[List[Optional[str]]] = None) -> Dict[str, float]:
    return {
        "EM": batch_metrics(preds, refs)['EM'],
        "TES": batch_metrics(preds, refs)['TES'],
        "EC": equational_consistency(preds, refs, group_ids),
        "RC": reasoning_consistency(preds, refs, group_ids),
    }

