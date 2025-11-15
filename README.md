# InThon 2025 데이터톤 트랙 - Baseline 모델

이 리포지토리는 **InThon 2025 데이터톤 트랙**의 베이스라인 구현입니다.

## 📁 프로젝트 구조

```
.
├── model.py              # 평가 대상 모델 (BaseModel 상속, predict 메서드 구현)
├── train.py              # 훈련 스크립트
├── dataloader.py         # 데이터셋 및 DataLoader
├── config.py             # 훈련/모델/토크나이저 설정
├── local_test.py         # 로컬 검증 스크립트
├── best_model.pt         # 학습된 모델 체크포인트 (Git LFS)
└── do_not_edit/          # ⚠️ 수정 금지: 평가용 검증 코드
    ├── model_template.py # BaseModel 인터페이스 정의
    ├── dataloader_validator.py  # 훈련 데이터 규정 검증기
    └── metric.py         # 평가 지표 (EM, TES)
```


### 🐳 로컬 개발 환경 (Docker 이미지 제공)

**평가 서버와 동일한 환경**에서 개발할 수 있도록 Docker 이미지를 제공합니다!

```bash
# 이미지 다운로드 (최초 1회)
docker pull jsh0423/pytorch-cuda:12.1

# 기본 사용 (Linux/WSL2 - GPU 지원)
docker run --gpus all -it --rm -v $(pwd):/workspace jsh0423/pytorch-cuda:12.1
```

**포함된 라이브러리:**
- PyTorch 2.5.1 (CUDA 12.1), NumPy, tqdm
- Transformers, Tokenizers, SentencePiece
- Accelerate, einops, safetensors
- Jupyter, Matplotlib, Pandas

**참고:** 이 이미지는 평가 환경과 동일한 라이브러리를 포함하므로, 로컬에서 정상 동작하면 서버에서도 동일하게 작동합니다.


### 로컬 검증

```bash
python local_test.py .
```

`local_test.py`는 다음을 검증합니다:
- Model 클래스 존재 및 BaseModel 상속
- predict 메서드 구현
- GPU 사용 여부
- 실행 시간 측정 및 전체 평가 시간 예측
- 상대 경로 사용 여부

### 4. 모델 사용
평가 서버는 모델을 다음과 같은 방식으로 불러옵니다.

```python
from model import Model

model = Model()
result = model.predict("123+456")
print(result)  # "579"
```


## 📚 참고 자료

- 대회 공식 플랫폼: https://jolly-bush-05d5dc000.3.azurestaticapps.net/index.html
- 대회 규칙 및 평가 가이드: https://jolly-bush-05d5dc000.3.azurestaticapps.net/rule.html

---

**Good luck! 🚀**
