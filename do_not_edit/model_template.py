"""
Model template for InThon Datathon.

This template provides a basic interface for participants to implement their models.
Participants should inherit from this class and implement the `predict` method.
The `predict` method should take a string input and return a string output.

"""

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass

import torch


class BaseModel(ABC):
    """참가자 모델의 기본 인터페이스
    
    참가자는 이 클래스를 상속하여 Model 클래스를 구현해야 합니다.
    __init__ 메서드는 인자를 받지 않거나, 모든 인자에 기본값이 설정되어 있어야 합니다.
    즉, Model()과 같이 인자 없이 호출 가능해야 합니다.
    """
    
    @abstractmethod
    def predict(self, input_text: str) -> str:
        """입력 문자열에 대한 예측을 반환
        
        Args:
            input_text: "12+34" 형식의 입력 문자열
        
        Returns:
            예측 결과 문자열 (반드시 문자열을 반환해야 함)
        """
        pass
    
    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보를 반환 (선택 사항)
        
        Returns:
            모델 메타데이터를 담은 딕셔너리
        """
        return {"model_type": self.__class__.__name__}

