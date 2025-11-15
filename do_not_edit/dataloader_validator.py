# --------------------------------------------------------------------------------------
# ⚠️ 편집 금지: 훈련 데이터의 "형식"과 "규정 준수"를 강제하는 검증기 (규칙 제3조 ②항)
# 
# 주요 검증 항목:
# - 입력/출력 형식 (문자열, 허용된 문자만 사용)
# - 괄호 균형 검사
# - 연산자 사용 규칙 (+, -, *, // 만 허용)
# - **자릿수 제한** (훈련 시 1~5자리만 허용, 규칙 제3조 ②항)
# 
# 이 검증기를 사용하지 않았을 때의 책임은 참가자에게 있습니다.
# --------------------------------------------------------------------------------------
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Dict, Any, Optional
import re

# ===== 규정 파라미터 (필요 시 이 부분만 수정 가능) =======================================
ALLOWED_DIGITS = set("0123456789")
ALLOWED_PARENS = {"(", ")"}
# [수정] 규정에 맞는 연산자 집합: '-' 추가
ALLOWED_OPERATORS = {"+", "*", "//", "-"}

# [수정] input_text에 허용되는 전체 문자 집합(문자 단위 검사용): '-' 추가
_ALLOWED_CHARS = ALLOWED_DIGITS | set("()+* /") | {"-"}

# [수정] 토큰화: 숫자, //, +, *, -, (, )
_TOKEN_RE = re.compile(r"(\d+|//|\+|\-|\*|\(|\))")

# 숫자 전용(정답/예측용) - 변경 없음 (음수 비허용)
_NUMERIC_ONLY_RE = re.compile(r"^[0-9]+$")

# **훈련 데이터 자릿수 제한** (규칙 제3조 ②항)
# 훈련 시: 1~5자리 숫자만 허용
# 평가 시: 6자리 이상도 허용
MAX_DIGITS_TRAINING = 5
MIN_DIGITS_TRAINING = 1


@dataclass
class ValidationStats:
    total_samples: int = 0
    invalid_input_chars: int = 0
    invalid_parentheses: int = 0
    invalid_operator_usage: int = 0
    empty_or_non_numeric_target: int = 0
    invalid_digit_count: int = 0  # 자릿수 제한 위반

    def to_dict(self) -> Dict[str, int]:
        return self.__dict__.copy()


class DatasetValidationError(ValueError):
    """훈련 데이터 규정 위반 시 발생하는 예외.
    
    규칙 제3조 ②항을 준수하지 않을 경우 발생합니다.
    """
    pass


def _check_chars_allowed(s: str) -> bool:
    # 문자 레벨 필터(빠른 사전 검사)
    return all(c in _ALLOWED_CHARS for c in s)


def _balanced_parentheses(s: str) -> bool:
    bal = 0
    for ch in s:
        if ch == "(":
            bal += 1
        elif ch == ")":
            bal -= 1
            if bal < 0:
                return False
    return bal == 0


def _tokens(s: str) -> List[str]:
    # 공백 제거 금지: 공백 자체도 허용 문자 아님(사전 검사에서 걸러짐).
    # 정규식 토큰화로 '//'를 한 토큰으로 유지.
    toks = _TOKEN_RE.findall(s)
    # 모든 문자 사용이 토큰으로 소진되었는지 점검
    if "".join(toks) != s:
        # 공백/허용 외 문자가 섞였거나, 미지원 토큰 패턴인 경우
        raise DatasetValidationError("Input contains invalid characters or unsupported tokens.")
    return toks


def _validate_digit_count(toks: List[str], is_training: bool = True) -> None:
    """
    훈련 데이터의 자릿수 제한 검증 (규칙 제3조 ②항).
    
    - 훈련 모드(is_training=True): 1~5자리 숫자만 허용
    - 평가 모드(is_training=False): 6자리 이상도 허용
    
    Args:
        toks: 토큰화된 입력 문자열
        is_training: 훈련 모드 여부 (기본값 True)
    
    Raises:
        DatasetValidationError: 자릿수 제한 위반 시
    """
    if not is_training:
        # 평가 모드에서는 자릿수 제한 없음
        return
    
    for tok in toks:
        if tok.isdigit():
            num_digits = len(tok)
            # 선행 0 제거 후 자릿수 계산 (예: "00123" -> "123" -> 3자리)
            num_value = int(tok)
            actual_digits = len(str(num_value)) if num_value > 0 else 1
            
            if actual_digits < MIN_DIGITS_TRAINING or actual_digits > MAX_DIGITS_TRAINING:
                raise DatasetValidationError(
                    f"훈련 데이터의 숫자는 {MIN_DIGITS_TRAINING}~{MAX_DIGITS_TRAINING}자리여야 합니다. "
                    f"발견된 숫자: {tok} ({actual_digits}자리). "
                    f"패딩(예: '00123')이나 {MAX_DIGITS_TRAINING}자리 초과 숫자 사용은 금지됩니다."
                )


def _validate_operator_usage(toks: List[str]) -> None:
    # 1) 단일 '/' 금지: 정규식으로는 애초에 생성되지 않지만, 방어적으로 확인
    if any("/" in t and t != "//" for t in toks):
        raise DatasetValidationError("Only '//' is allowed; single '/' is not permitted.")

    # 2) 허용된 연산자 외 금지
    for t in toks:
        # [수정] '-' 연산자 검사 추가
        if t in {"+", "*", "//", "-"} and t not in ALLOWED_OPERATORS:
            raise DatasetValidationError(f"Operator '{t}' not allowed by configuration.")

    # 3) 토큰 시퀀스의 최소 문법 점검(아주 간단한 수준)
    #    - 숫자/')' 뒤에는 연산자/')' 끝이어야 함
    #    - '(' 뒤에는 숫자/'(' 이어야 함
    prev = None
    for t in toks:
        if prev is None:
            # [수정] '-' 추가
            if t in {"+", "*", "//", "-", ")"}:
                raise DatasetValidationError("Expression cannot start with operator or ')'.")
        else:
            # [수정] '-' 추가
            if prev in {"+", "*", "//", "(", "-"}:
                if t in {"+", "*", "//", "-", ")"}:
                    raise DatasetValidationError("Two operators or illegal token order.")
            elif prev.isdigit() or prev == ")":
                if t.isdigit() or t == "(":
                    raise DatasetValidationError("Missing operator between terms.")
        prev = t
    # [수정] '-' 추가
    if toks and toks[-1] in {"+", "*", "//", "(", "-"}:
        raise DatasetValidationError("Expression cannot end with operator or '('.")

    # 4) 즉시 0으로 나누기 방지(보수적 체크): '//' 바로 다음이 '0' 상수일 경우만 차단
    for i, t in enumerate(toks[:-1]):
        if t == "//" and toks[i + 1].isdigit() and int(toks[i + 1]) == 0:
            raise DatasetValidationError("Division by zero is not allowed.")


def validate_sample(sample: Dict[str, Any], stats: Optional[ValidationStats] = None, is_training: bool = True) -> None:
    """
    샘플 규정 준수 점검. 위반 시 DatasetValidationError 발생.
    
    Args:
        sample: 검증할 샘플 {'input_text': str, 'target_text': str, ...}
        stats: 통계 기록용 (선택)
        is_training: 훈련 모드 여부 (기본값 True)
                     - True: 1~5자리 숫자만 허용 (규칙 제3조 ②항)
                     - False: 6자리 이상도 허용 (제출 전 평가 시)
    
    Raises:
        DatasetValidationError: 규정 위반 시
    """
    if "input_text" not in sample or "target_text" not in sample:
        raise DatasetValidationError("Sample must contain 'input_text' and 'target_text' keys.")

    input_text = sample["input_text"]
    target_text = sample["target_text"]

    if stats:
        stats.total_samples += 1

    # ---- input_text 검사
    if not isinstance(input_text, str) or not input_text:
        if stats:
            stats.invalid_input_chars += 1
        raise DatasetValidationError("input_text must be a non-empty string.")

    if not _check_chars_allowed(input_text):
        if stats:
            stats.invalid_input_chars += 1
        raise DatasetValidationError("input_text contains disallowed characters.")

    if not _balanced_parentheses(input_text):
        if stats:
            stats.invalid_parentheses += 1
        raise DatasetValidationError("input_text has unbalanced parentheses.")

    toks = _tokens(input_text)  # invalid char/unknown token caught here
    try:
        _validate_operator_usage(toks)
    except DatasetValidationError:
        if stats:
            stats.invalid_operator_usage += 1
        raise
    
    # **자릿수 제한 검증** (규칙 제3조 ②항)
    try:
        _validate_digit_count(toks, is_training=is_training)
    except DatasetValidationError:
        if stats:
            stats.invalid_digit_count += 1
        raise

    # ---- target_text 검사 (숫자 전용) - 변경 없음 (음수 비허용)
    if not isinstance(target_text, str) or not target_text or not _NUMERIC_ONLY_RE.match(target_text):
        if stats:
            stats.empty_or_non_numeric_target += 1
        raise DatasetValidationError("target_text must be a non-empty numeric string (digits only).")


def collate_fn_with_validation(samples: List[Dict[str, Any]], is_training: bool = True) -> Dict[str, Any]:
    """
    규정 준수를 강제하는 collate_fn.
    
    Args:
        samples: 배치에 포함될 샘플 리스트
        is_training: 훈련 모드 여부 (자릿수 제한 검증에 사용)
    
    배치 형식:
    {
        "input_text": List[str],
        "target_text": List[str],
        "meta": Optional[List[dict]]
    }
    """
    stats = ValidationStats()
    for s in samples:
        validate_sample(s, stats, is_training=is_training)

    batch = {
        "input_text": [s["input_text"] for s in samples],
        "target_text": [s["target_text"] for s in samples],
    }
    # meta는 선택(optional)
    if any("meta" in s for s in samples):
        batch["meta"] = [s.get("meta", {}) for s in samples]
    return batch


class DatasetGuard(Iterable):
    """
    IterableDataset 래퍼(게으른 검증). __iter__ 단계에서 샘플을 하나씩 검증.
    
    Args:
        iterable: 검증할 샘플을 생성하는 iterable
        is_training: 훈련 모드 여부 (자릿수 제한 검증에 사용, 기본값 True)
    """
    def __init__(self, iterable: Iterable[Dict[str, Any]], is_training: bool = True):
        self._iterable = iterable
        self._is_training = is_training

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        stats = ValidationStats()
        for s in self._iterable:
            validate_sample(s, stats, is_training=self._is_training)
            yield s
