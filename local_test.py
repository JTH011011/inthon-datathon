import sys

import os

from pathlib import Path

import importlib.util

import time

import torch

import tempfile

import shutil


def get_all_devices_from_model(model) -> set:
    """
    모델의 모든 파라미터와 버퍼를 재귀적으로 탐색하여 실제 device를 확인
    
    모델 구조가 자유롭게 바뀔 수 있으므로, 모든 속성을 재귀적으로 탐색합니다.
    
    Returns:
        발견된 모든 device의 집합
    """
    devices = set()
    visited = set()  # 순환 참조 방지
    
    def traverse(obj, depth=0):
        """재귀적으로 모든 텐서를 탐색"""
        if depth > 30:  # 깊이 제한
            return
        
        # 순환 참조 방지 (id 기반)
        try:
            obj_id = id(obj)
            if obj_id in visited:
                return
            visited.add(obj_id)
        except TypeError:
            # id()를 사용할 수 없는 객체 (예: None, 숫자 등)
            pass
        
        # PyTorch 모듈인 경우
        if isinstance(obj, torch.nn.Module):
            try:
                # 모든 파라미터 확인
                for param in obj.parameters(recurse=False):  # 현재 모듈만
                    if param.is_cuda:
                        devices.add(f"cuda:{param.device.index}" if param.device.index is not None else "cuda")
                    else:
                        devices.add("cpu")
                
                # 모든 버퍼 확인
                for buffer in obj.buffers(recurse=False):  # 현재 모듈만
                    if buffer.is_cuda:
                        devices.add(f"cuda:{buffer.device.index}" if buffer.device.index is not None else "cuda")
                    else:
                        devices.add("cpu")
                
                # 모든 하위 모듈 재귀 탐색
                for name, child in obj.named_children():
                    traverse(child, depth + 1)
            except Exception:
                pass  # 모듈 탐색 실패 시 무시
        
        # 텐서인 경우
        elif isinstance(obj, torch.Tensor):
            try:
                if obj.is_cuda:
                    devices.add(f"cuda:{obj.device.index}" if obj.device.index is not None else "cuda")
                else:
                    devices.add("cpu")
            except Exception:
                pass
        
        # 딕셔너리인 경우
        elif isinstance(obj, dict):
            for value in obj.values():
                traverse(value, depth + 1)
        
        # 리스트나 튜플인 경우
        elif isinstance(obj, (list, tuple)):
            for item in obj:
                traverse(item, depth + 1)
        
        # 일반 객체인 경우 - 주요 속성들 확인 (깊이 제한)
        elif depth < 10 and hasattr(obj, '__dict__'):
            try:
                # __dict__의 값들만 확인 (메서드 호출 최소화)
                if hasattr(obj, '__dict__'):
                    for value in obj.__dict__.values():
                        traverse(value, depth + 1)
            except Exception:
                pass
    
    # 모델 자체부터 시작
    traverse(model)
    
    # model.model 같은 중첩 구조도 명시적으로 확인
    if hasattr(model, 'model'):
        traverse(model.model)
    
    return devices


def run_local_test(submission_dir_path: str):
    submission_path = Path(submission_dir_path).resolve()
    model_py_path = submission_path / "model.py"

    if not model_py_path.exists():
        print(f"오류: {submission_path}에서 model.py를 찾을 수 없습니다.")
        return

    original_cwd = os.getcwd()
    os.chdir(submission_path)

    try:
        spec = importlib.util.spec_from_file_location("submission_main", model_py_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        spec.loader.exec_module(module)
        # Model 클래스 존재 여부

        if not hasattr(module, "Model"):
            print("❌ model.py 안에 'Model' 클래스가 없습니다.")
            return

        from do_not_edit.model_template import BaseModel
        ModelClass = getattr(module, "Model")
        if not issubclass(ModelClass, BaseModel):
            print("❌ Model 클래스가 BaseModel을 상속하지 않습니다.")
            return

        # 초기화 테스트

        try:
            model = ModelClass()

        except TypeError as e:
            print(f"❌ Model() 초기화 실패 (인자 관련 문제 가능): {e}")
            return

        print("✅ Model 초기화 완료.")
        
        # 상대 경로 검증 (실제 디렉토리 변경 테스트)
        print("\n[상대 경로 검증]")
        print("   다른 디렉토리에서 모델 초기화 테스트 중...")
        
        # 임시 디렉토리 생성
        temp_dir = tempfile.mkdtemp(prefix="inthon_test_")
        temp_path = Path(temp_dir)
        
        try:
            # 필요한 파일들을 임시 디렉토리로 복사
            # model.py는 이미 로드되었으므로 복사만 하면 됨
            files_to_copy = [
                "model.py",
                "best_model.pt",
                "do_not_edit",
            ]
            
            # 추가로 필요한 파일들도 복사 (import 오류 방지)
            optional_files = [
                "config.py",
                "dataloader.py",
            ]
            
            copied_files = []
            for file_name in files_to_copy:
                src = submission_path / file_name
                if src.exists():
                    if src.is_dir():
                        shutil.copytree(src, temp_path / file_name)
                    else:
                        shutil.copy2(src, temp_path / file_name)
                    copied_files.append(file_name)
            
            for file_name in optional_files:
                src = submission_path / file_name
                if src.exists():
                    shutil.copy2(src, temp_path / file_name)
            
            # best_model.pt가 없으면 경고
            if "best_model.pt" not in copied_files:
                print("   ⚠️  best_model.pt가 없어 상대 경로 검증을 건너뜁니다.")
                print("   ⚠️  (모델 초기화는 성공했지만, 다른 디렉토리 테스트는 불가능)")
            else:
                # 임시 디렉토리로 이동하여 모델 초기화 시도
                original_cwd_for_test = os.getcwd()
                os.chdir(temp_path)
                
                try:
                    # 모듈을 다시 로드 (새 디렉토리에서)
                    spec = importlib.util.spec_from_file_location("submission_test", temp_path / "model.py")
                    test_module = importlib.util.module_from_spec(spec)
                    assert spec is not None and spec.loader is not None
                    spec.loader.exec_module(test_module)
                    
                    # 새 디렉토리에서 모델 초기화 시도
                    TestModelClass = getattr(test_module, "Model")
                    test_model = TestModelClass()
                    
                    # predict 테스트
                    test_result = test_model.predict("12+34")
                    
                    if isinstance(test_result, str):
                        print("✅ 상대 경로 사용 확인됨 (다른 디렉토리에서도 정상 작동)")
                    else:
                        print("❌ 상대 경로 검증 실패: predict() 반환 타입 오류")
                        
                except FileNotFoundError as e:
                    print(f"❌ 상대 경로 검증 실패: 파일을 찾을 수 없음")
                    print(f"   오류: {e}")
                    print("   ⚠️  절대 경로를 사용하고 있을 가능성이 있습니다 (규칙 제9조 ③항)")
                except Exception as e:
                    print(f"❌ 상대 경로 검증 실패: 다른 디렉토리에서 모델 초기화 실패")
                    print(f"   오류: {e}")
                    print("   ⚠️  절대 경로를 사용하고 있을 가능성이 있습니다 (규칙 제9조 ③항)")
                finally:
                    os.chdir(original_cwd_for_test)
                    
        finally:
            # 임시 디렉토리 정리
            try:
                shutil.rmtree(temp_dir)
            except Exception:
                pass  # 정리 실패해도 무시
        
        # GPU 사용 여부 체크 (모델의 모든 파라미터를 재귀적으로 탐색)
        print("\n[GPU 사용 여부 검증]")
        
        # 모델의 모든 파라미터와 버퍼를 재귀적으로 탐색하여 실제 device 확인
        devices = get_all_devices_from_model(model)
        
        if not devices:
            print("⚠️  모델에서 파라미터나 텐서를 찾을 수 없습니다.")
        else:
            # CPU와 CUDA device 분리
            cuda_devices = {d for d in devices if d.startswith("cuda")}
            cpu_devices = {d for d in devices if d == "cpu"}
            
            if cuda_devices and cpu_devices:
                print("⚠️  경고: 모델의 일부는 GPU에, 일부는 CPU에 있습니다.")
                print(f"   GPU device: {', '.join(sorted(cuda_devices))}")
                print(f"   CPU device: {', '.join(sorted(cpu_devices))}")
                print("   ⚠️  모든 파라미터가 같은 device에 있어야 합니다.")
            elif cuda_devices:
                # GPU 사용 중
                gpu_device = sorted(cuda_devices)[0]
                if torch.cuda.is_available():
                    device_index = int(gpu_device.split(":")[1]) if ":" in gpu_device else 0
                    gpu_name = torch.cuda.get_device_name(device_index)
                    print(f"✅ GPU 사용 중: {gpu_device}")
                    print(f"   GPU 이름: {gpu_name}")
                    if len(cuda_devices) > 1:
                        print(f"   ⚠️  여러 GPU device 발견: {', '.join(sorted(cuda_devices))}")
                else:
                    print(f"⚠️  경고: CUDA가 사용 불가능한데 GPU 디바이스로 설정되어 있습니다.")
                    print(f"   발견된 device: {', '.join(sorted(cuda_devices))}")
            elif cpu_devices:
                print("⚠️  CPU 사용 중")
                print("   ⚠️  GPU가 사용 가능한 경우 GPU를 사용하는 것을 권장합니다.")
            else:
                print(f"⚠️  알 수 없는 device: {devices}")
        
        # device 속성도 확인 (참고용)
        if hasattr(model, "device"):
            device_attr = model.device
            if isinstance(device_attr, torch.device):
                print(f"   (참고: model.device = {device_attr})")
        
        # predict 테스트
        if not hasattr(model, "predict"):
            print("❌ Model에 predict 메서드가 없습니다.")
            return

        test_input = "12+34"
        print(f"\n[predict 테스트] 입력='{test_input}'")

        # 실행 시간 측정
        try:
            # 워밍업 (첫 실행은 느릴 수 있음)
            _ = model.predict(test_input)
            
            # 실제 시간 측정
            num_runs = 10  # 여러 번 실행하여 평균 측정
            times = []
            for _ in range(num_runs):
                start_time = time.perf_counter()
                pred = model.predict(test_input)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            
            avg_time = sum(times) / len(times)
            min_time = min(times)
            max_time = max(times)
            
        except Exception as e:
            print(f"❌ predict 호출 중 예외: {e}")
            return

        if not isinstance(pred, str):
            print(f"❌ predict()는 반드시 문자열을 반환해야 합니다. 현재 타입: {type(pred)}")
            return
        
        print(f"✅ predict() 성공, 반환값: '{pred}' (type={type(pred)})")
        
        # 실행 시간 정보 출력
        print(f"\n[실행 시간 측정]")
        print(f"   평균 실행 시간: {avg_time*1000:.2f}ms ({avg_time:.4f}초)")
        print(f"   최소 실행 시간: {min_time*1000:.2f}ms ({min_time:.4f}초)")
        print(f"   최대 실행 시간: {max_time*1000:.2f}ms ({max_time:.4f}초)")
        
        # 전체 평가 시간 예측 (10000개 샘플 기준)
        total_samples = 10000
        server_setup_time_seconds = 3 * 60  # 서버 세팅 시간 3분
        inference_time_seconds = avg_time * total_samples
        estimated_total_seconds = server_setup_time_seconds + inference_time_seconds
        estimated_minutes = estimated_total_seconds / 60
        estimated_hours = estimated_minutes / 60
        
        print(f"\n[전체 평가 시간 예측] (10000개 샘플 기준)")
        print(f"   서버 세팅 시간: {server_setup_time_seconds/60:.0f}분")
        print(f"   추론 시간: {inference_time_seconds:.1f}초 ({inference_time_seconds/60:.2f}분)")
        print(f"   총 예상 시간: ", end="")
        if estimated_minutes < 1:
            print(f"{estimated_total_seconds:.1f}초")
        elif estimated_hours < 1:
            print(f"{estimated_minutes:.1f}분 ({estimated_total_seconds:.0f}초)")
        else:
            print(f"{estimated_hours:.2f}시간 ({estimated_minutes:.1f}분)")
        
        print("\n🎉 local_test 통과! InThon 규정 형식 OK.")
    finally:
        os.chdir(original_cwd)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("사용법: python local_test.py <제출 디렉토리 경로>")
    else:

        run_local_test(sys.argv[1])

