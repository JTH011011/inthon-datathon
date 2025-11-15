"""베이스라인 데이터 생성 및 DataLoader"""
from __future__ import annotations
from typing import Dict, Any, Tuple, Union
import random
from functools import partial
from torch.utils.data import DataLoader, Dataset

from do_not_edit.dataloader_validator import collate_fn_with_validation


def _rand_int(rng: random.Random, num_digits: Tuple[int, int]) -> int:
    """랜덤 정수 생성 (선행 0 없음)"""
    lo, hi = num_digits
    n = rng.randint(lo, hi)
    if n == 1:
        return rng.randint(0, 9)
    first = rng.randint(1, 9)
    rest = [rng.randint(0, 9) for _ in range(n - 1)]
    return int(str(first) + "".join(str(x) for x in rest))


def _gen_expr(rng: random.Random, depth: int, num_digits: Tuple[int, int]) -> Tuple[str, int]:
    """재귀적 수식 생성 (연산자 우선순위 반영)"""
    
    def number() -> Tuple[str, int]:
        v = _rand_int(rng, num_digits)
        return str(v), v
    
    def factor(d: int) -> Tuple[str, int]:
        if d == 0 or rng.random() < 0.7:
            return number()
        e, v = expr(d - 1)
        return f"({e})", v
    
    def term(d: int) -> Tuple[str, int]:
        if d == 0 or rng.random() < 0.5:
            return factor(d)
        
        op = rng.choice(["*", "//"])
        le, lv = term(rng.randint(0, d - 1))
        re, rv = factor(rng.randint(0, d - 1))
        
        if op == "//" and rv == 0:
            rv = rng.randint(1, 9)
            re = str(rv)
        
        v = lv // rv if op == "//" else lv * rv
        return f"{le}{op}{re}", v
    
    def expr(d: int) -> Tuple[str, int]:
        if d == 0 or rng.random() < 0.5:
            return term(d)
        
        op = rng.choice(["+", "-"])
        le, lv = expr(rng.randint(0, d - 1))
        re, rv = term(rng.randint(0, d - 1))
        
        # 음수 방지
        if op == "-" and lv < rv:
            lv, rv = rv, lv
            le, re = re, le
        
        v = lv + rv if op == "+" else lv - rv
        return f"{le}{op}{re}", v
    
    return expr(depth)


def _swap_commutative(expr: str, val: int) -> Tuple[str, int]:
    """
    수식의 최상위 연산자가 + 또는 *일 경우, 좌우 피연산자를 뒤바꿉니다.
    예: "1+2*3" -> "2*3+1"
    주의: 매우 단순한 구현이며, 복잡한 괄호 구조에서는 실패할 수 있습니다.
    """
    
    # 괄호 깊이를 추적하여 최상위 연산자 찾기
    paren_depth = 0
    split_idx = -1
    op = ""

    # 뒤에서부터 순회하며 첫 번째 최상위 +, - 연산자 찾기
    for i in range(len(expr) - 1, -1, -1):
        char = expr[i]
        if char == ")":
            paren_depth += 1
        elif char == "(":
            paren_depth -= 1
        elif paren_depth == 0 and char in "+-":
            split_idx = i
            op = char
            break
    
    # +, -가 없으면, *, // 연산자 찾기
    if split_idx == -1:
        for i in range(len(expr) - 1, -1, -1):
            char = expr[i]
            if char == ")":
                paren_depth += 1
            elif char == "(":
                paren_depth -= 1
            elif paren_depth == 0 and char in "*/":
                # 곱셈/나눗셈은 연달아 나올 수 있으므로, 맨 앞 연산자를 찾는다.
                if i > 0 and expr[i-1] in "*/": # `//` 처리
                    continue
                split_idx = i
                op = expr[i-1:i+1] if expr[i-1] == "/" else char # `//` 처리
                break

    # 교환 가능한 연산자일 경우 좌우를 바꿈
    if op in "+*" and split_idx != -1:
        left = expr[:split_idx]
        right = expr[split_idx + len(op):]
        
        # // 연산자 처리
        if op == "*" and expr[split_idx-1:split_idx+1] == "//":
            return expr, val

        return f"{right}{op}{left}", val
        
    return expr, val


class ArithmeticDataset(Dataset):
    """사칙연산 수식 생성 Dataset"""
    
    def __init__(
        self,
        num_samples: int,
        max_depth: int = 2,
        num_digits: Tuple[int, int] = (1, 3),
        seed: int = 42,
        mode: str = "train",
        augment_commutative_prob: float = 0.0,
    ):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.num_digits = num_digits
        self.seed = seed
        self.mode = mode
        self.augment_commutative_prob = augment_commutative_prob
    
    def __len__(self) -> int:
        return self.num_samples
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        rng = random.Random(self.seed + idx)
        depth = rng.randint(0, self.max_depth)
        expr, val = _gen_expr(rng, depth, self.num_digits)

        # Commutative augmentation
        if self.mode == "train" and rng.random() < self.augment_commutative_prob:
            expr, val = _swap_commutative(expr, val)

        return {"input_text": expr, "target_text": str(val), "meta": {"depth": depth}}


def get_dataloader(
    dataset: Dataset,
    batch_size: int = 64,
    num_workers: int = 0,
    pin_memory: bool = False,
    mode: str = "train",
) -> DataLoader:
    """DataLoader 생성 (mode="train"일 때 자동 검증)"""
    is_training = (getattr(dataset, "mode", mode) == "train")
    collate_fn = partial(collate_fn_with_validation, is_training=is_training) if is_training else None
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
    )
