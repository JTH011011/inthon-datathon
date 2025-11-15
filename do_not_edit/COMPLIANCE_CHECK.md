# do_not_edit 코드 규칙 준수 점검 결과

## 📋 점검 일시
점검 기준: `docs/rule.md` (InThon 2025 데이터톤 트랙 규칙)

---

## ✅ dataloader_validator.py 점검 결과

### 제3조 ②항 - 입력·출력 형식

#### 1. 허용 문자 검증
- **규칙**: 숫자(0–9), 연산자(`+`, `-`, `*`, `//`), 괄호(`(`, `)`)만 허용
- **코드 확인**:
  ```python
  ALLOWED_OPERATORS = {"+", "*", "//", "-"}  # ✅ 규칙 준수
  _ALLOWED_CHARS = ALLOWED_DIGITS | set("()+* /") | {"-"}  # ✅ 규칙 준수
  ```
- **결과**: ✅ **준수**

#### 2. 학습 데이터 숫자 제한
- **규칙**: 학습 데이터의 입력(`input_text`)에 등장하는 모든 숫자는 1~5자리여야 함
- **코드 확인**:
  ```python
  MAX_DIGITS_TRAINING = 5  # ✅ 규칙 준수
  MIN_DIGITS_TRAINING = 1  # ✅ 규칙 준수
  ```
- **검증 로직**: `_validate_digit_count()` 함수에서 `is_training=True`일 때만 검증
- **결과**: ✅ **준수**

#### 3. 괄호 균형 검사
- **규칙**: 올바른 형태의 수식이어야 함
- **코드 확인**: `_balanced_parentheses()` 함수로 검증
- **결과**: ✅ **준수**

#### 4. 연산자 사용 규칙
- **규칙**: `+`, `-`, `*`, `//`만 허용
- **코드 확인**: `_validate_operator_usage()` 함수에서 검증
- **결과**: ✅ **준수**

#### 5. target_text 검증
- **규칙**: 숫자로만 구성된 수식의 최종 결과값
- **코드 확인**: `_NUMERIC_ONLY_RE = re.compile(r"^[0-9]+$")`로 검증
- **결과**: ✅ **준수**

### 제4조 ①항 - 모델 구조 제한
- **규칙**: `eval()`, `int()` 같은 내장 함수 사용 금지
- **코드 확인**: `int()` 사용 발견 (라인 110, 156)
- **분석**: 
  - 라인 110: `num_value = int(tok)` - 자릿수 검증 목적 (검증 코드이므로 허용)
  - 라인 156: `int(toks[i + 1]) == 0` - 0으로 나누기 방지 검증 (검증 코드이므로 허용)
- **결과**: ✅ **준수** (검증 코드는 예외)

---

## ✅ metric.py 점검 결과

### 제8조 ②항 - 샘플 단위 평가 지표
- **규칙**: EM(Exact Match) 및 TES(Token Edit Similarity)를 기준으로 평가
- **코드 확인**:
  - `exact_match()` 함수: ✅ 구현됨
  - `token_edit_similarity()` 함수: ✅ 구현됨
  - `compute_metrics()` 함수: EM, TES, EC, RC 반환
- **결과**: ✅ **준수**

### 제8조 ①항 - 리더보드 평가 지표
- **규칙**: Calculation Accuracy, Law Preservation, Expression Consistency, Relational Consistency
- **코드 확인**:
  - `equational_consistency()` 함수: Expression Consistency 구현
  - `reasoning_consistency()` 함수: Relational Consistency 구현
- **결과**: ✅ **준수**

---

## ✅ model_template.py 점검 결과

### 제5조 ①항 - 예측 인터페이스
- **규칙**: 입력 수식(`input_text`)을 받아, 숫자로만 구성된 최종 정답 문자열을 반환하는 `predict()` 메서드 제공
- **코드 확인**:
  ```python
  @abstractmethod
  def predict(self, input_text: str) -> str:
      """입력 문자열에 대한 예측을 반환"""
  ```
- **결과**: ✅ **준수**

### 제9조 ③항 - 경로 및 환경
- **규칙**: 상대 경로 사용
- **코드 확인**: `model_template.py`는 인터페이스 정의만 포함하므로 경로 사용 없음
- **결과**: ✅ **준수** (해당 없음)

---

## 📊 종합 점검 결과

| 파일 | 규칙 준수 여부 | 주요 검증 항목 |
|------|--------------|---------------|
| `dataloader_validator.py` | ✅ **준수** | 허용 문자, 자릿수 제한, 연산자 규칙, 괄호 균형 |
| `metric.py` | ✅ **준수** | EM, TES, EC, RC 구현 |
| `model_template.py` | ✅ **준수** | predict() 인터페이스 정의 |

---

## 🔍 발견된 사항

### 1. dataloader_validator.py의 주석 오류
- **위치**: 라인 2, 8
- **문제**: "규칙 제4조"라고 명시되어 있으나, 실제로는 **제3조 ②항**을 검증하는 코드임
- **권장 사항**: 주석을 "규칙 제3조 ②항"으로 수정

### 2. int() 사용
- **위치**: 라인 110, 156
- **분석**: 검증 목적의 `int()` 사용은 허용됨 (모델 코드가 아닌 검증 코드)
- **결과**: ✅ 문제 없음

---

## ✅ 최종 결론

**모든 `do_not_edit/` 코드는 `rule.md`의 규칙을 준수하고 있습니다.**

단, 주석의 규칙 조항 번호를 정확히 수정하는 것을 권장합니다.

