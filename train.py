from __future__ import annotations
from typing import List, Any, Tuple
from config import TrainConfig, ModelConfig, TokenizerConfig

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import (
    ArithmeticDataset,  # 사칙연산 데이터를 만들어주는 Dataset
    get_dataloader,     # Dataset을 받아서 DataLoader로 바꿔주는 함수
)
from do_not_edit.metric import compute_metrics  # EM, TES 같은 간단한 성능 지표

from model import (
    TransformerSeq2Seq,
    CharTokenizer,      # 문자 단위 토크나이저
    tokenize_batch,     # batch(dict)를 토크나이즈 + 패딩까지 해주는 함수
    INPUT_CHARS,        # 입력 문자 집합
    OUTPUT_CHARS,       # 출력 문자 집합
)

import os
import wandb


def load_checkpoint(
    ckpt_path: str,
    model: nn.Module,
    optim: torch.optim.Optimizer | None = None,
    scheduler: torch.optim.lr_scheduler._LRScheduler | None = None,
    device: torch.device | str = "cpu",
) -> int:
    """
    ckpt_path에서 체크포인트를 로드해서 model/optimizer/scheduler 상태를 복원합니다.
    반환값: 마지막 step (없으면 0)
    """
    if not ckpt_path or not os.path.exists(ckpt_path):
        print(f"[ckpt] No checkpoint found at {ckpt_path}, starting from scratch.")
        return 0

    print(f"[ckpt] Loading checkpoint from {ckpt_path} ...")
    ckpt = torch.load(ckpt_path, map_location=device)

    # 모델 파라미터 로드
    model.load_state_dict(ckpt["model_state"])
    print("[ckpt] Loaded model_state")

    # 옵티마이저 상태 로드 (있을 때만)
    if optim is not None and "optim_state" in ckpt:
        optim.load_state_dict(ckpt["optim_state"])
        print("[ckpt] Loaded optim_state")
    
    # 스케줄러 상태 로드 (있을 때만)
    if scheduler is not None and "scheduler_state" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler_state"])
        print("[ckpt] Loaded scheduler_state")

    start_step = int(ckpt.get("step", 0))
    print(f"[ckpt] Resuming from step {start_step}")
    return start_step


# ======================================================================================
# 1. 학습 루프
# ======================================================================================
def train_loop(
    model: nn.Module,           # 학습할 모델
    dataloader: DataLoader,     # DataLoader를 직접 전달받아 사용합니다.
    input_tokenizer: CharTokenizer,      # (tokenize_batch 유틸리티를 위해 유지)
    output_tokenizer: CharTokenizer,     # 출력 문자 토크나이저
    device: torch.device,       # cpu 또는 cuda
    val_dataloader: DataLoader | None = None,  # 별도 검증 DataLoader (None이면 훈련 배치로 검증)
    *,
    train_config: TrainConfig,  # 학습 설정
    model_config: ModelConfig,  # 모델 설정
    tokenizer_config: TokenizerConfig,  # 토크나이저 설정
):
    # 모델을 GPU/CPU로 보냄
    model.to(device)

    # 옵티마이저: AdamW는 Adam + weight decay가 들어간 버전
    optim = torch.optim.AdamW(model.parameters(), lr=train_config.lr)
    
    # LR 스케줄러: CosineAnnealingLR
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optim, T_max=train_config.max_train_steps or 100_000
    )

    # seq2seq에서 흔히 쓰는 CE loss
    # pad 토큰은 무시하도록(ignore_index) 설정
    loss_fn = nn.CrossEntropyLoss(ignore_index=output_tokenizer.pad_id)

    step = 0
    if train_config.save_best_path is not None:
        step = load_checkpoint(
            ckpt_path=train_config.save_best_path,
            model=model,
            optim=optim,
            scheduler=scheduler,
            device=device,
        )

    pbar = tqdm(
        initial=step,
        total=train_config.max_train_steps if train_config.max_train_steps is not None else None,
        desc="train",
        unit="step",
        ncols=120,
        dynamic_ncols=True,
        leave=True,
    )

    # best EM 추적용 변수 (None이 아니면 개선 시 모델 저장)
    best_em = float("-inf")
    model.train()

    for epoch in range(train_config.num_epochs):
        if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
            break
        pbar.write(f"Starting epoch {epoch + 1}/{train_config.num_epochs}")

        for batch in dataloader:
            batch_tensors = tokenize_batch(batch, input_tokenizer, output_tokenizer)
            src = batch_tensors.src.to(device)
            target_input = batch_tensors.tgt_inp.to(device)
            target_output = batch_tensors.tgt_out.to(device)

            logits = model(
                src,
                target_input,
                src_pad_id=input_tokenizer.pad_id,
                tgt_pad_id=output_tokenizer.pad_id,
            )

            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                target_output.view(-1),
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optim.step()
            scheduler.step()
            optim.zero_grad()

            step += 1
            pbar.update(1)

            if wandb.run is not None:
                wandb.log(
                    {
                        "train/loss": float(loss.item()),
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/epoch": epoch,
                    },
                    step=step,
                )

            if step % train_config.valid_every == 0:
                model.eval()
                with torch.no_grad():
                    preds_all: List[str] = []
                    targets_all: List[str] = []
                    inputs_all: List[str] = []

                    for val_batch in val_dataloader:
                        val_bt = tokenize_batch(val_batch, input_tokenizer, output_tokenizer)
                        val_src = val_bt.src.to(device)
                        gen_ids = model.generate(
                            src=val_src,
                            max_len=train_config.max_gen_len,
                            bos_id=output_tokenizer.bos_id,
                            eos_id=output_tokenizer.eos_id,
                            src_pad_id=input_tokenizer.pad_id,
                        )

                        for i in range(gen_ids.size(0)):
                            seq_chars: List[str] = []
                            for t in gen_ids[i].tolist():
                                idx = int(t)
                                if idx == output_tokenizer.eos_id:
                                    break
                                if idx in output_tokenizer.itos:
                                    ch = output_tokenizer.itos[idx]
                                    if ch.isdigit() or (ch == '-' and not seq_chars):
                                        seq_chars.append(ch)
                            pred_str = "".join(seq_chars)
                            if pred_str == "-":
                                pred_str = ""
                            preds_all.append(pred_str)
                        
                        targets_all.extend(val_batch["target_text"])
                        inputs_all.extend(val_batch["input_text"])

                    em_batch = compute_metrics(preds_all, targets_all)

                    pbar.write(f"[valid {step}] EM={em_batch['EM']:.3f} TES={em_batch['TES']:.3f}")
                    pbar.set_postfix(EM=f"{em_batch['EM']:.3f}", TES=f"{em_batch['TES']:.3f}")
                    pbar.refresh()

                    if wandb.run is not None:
                        wandb.log(
                            {
                                "valid/EM": float(em_batch["EM"]),
                                "valid/TES": float(em_batch["TES"]),
                            },
                            step=step,
                        )

                    if train_config.save_best_path is not None:
                        current_em = float(em_batch.get("EM", -1.0))
                        if current_em > best_em:
                            best_em = current_em
                            ckpt = {
                                "model_state": model.state_dict(),
                                "optim_state": optim.state_dict(),
                                "scheduler_state": scheduler.state_dict(),
                                "step": step,
                                "train_config": train_config.__dict__,
                                "model_config": model_config.__dict__,
                                "tokenizer_config": tokenizer_config.__dict__,
                            }
                            torch.save(ckpt, train_config.save_best_path)
                            pbar.write(
                                f"New best EM={best_em:.3f} at step {step}; "
                                f"saved to {train_config.save_best_path}"
                            )

                        if wandb.run is not None:
                            wandb.log({"valid/best_EM": best_em}, step=step)

                    n_show = min(train_config.show_valid_samples, len(preds_all))
                    pbar.write("Sample validation output:")
                    for i in range(n_show):
                        input_str = inputs_all[i]
                        tgt = targets_all[i]
                        pred = preds_all[i]
                        ok = "OK" if pred == tgt else "ERR"
                        pbar.write(f"  [{i}] {ok} | input: {input_str} | target: {tgt} | pred: {pred}")
                model.train()

            if train_config.max_train_steps is not None and step >= train_config.max_train_steps:
                break

# ======================================================================================
# 2. main 함수
# ======================================================================================
def main():
    # MPS (Apple Silicon GPU) 지원을 추가합니다.
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    train_dataset = ArithmeticDataset(
        num_samples=500_000,
        max_depth=3,
        num_digits=(1, 5),
        seed=123,
        mode="train",
        augment_commutative_prob=0.5,  # 50% 확률로 교환법칙 증강 적용
    )
    # MPS 사용 시 pin_memory=False로 설정해야 경고가 발생하지 않습니다.
    pin_memory = device.type != "mps"
    train_dataloader = get_dataloader(
        train_dataset, batch_size=256, num_workers=4, pin_memory=pin_memory
    )

    val_dataset = ArithmeticDataset(
        num_samples=1024, max_depth=4, num_digits=(1, 6), seed=999, mode="val"
    )
    val_dataloader = get_dataloader(
        val_dataset, batch_size=256, num_workers=4, pin_memory=pin_memory
    )

    tokenizer_config = TokenizerConfig(
        input_chars=INPUT_CHARS, output_chars=OUTPUT_CHARS, add_special=True
    )

    input_tokenizer = CharTokenizer(
        tokenizer_config.input_chars, add_special=tokenizer_config.add_special
    )
    output_tokenizer = CharTokenizer(
        tokenizer_config.output_chars, add_special=tokenizer_config.add_special
    )

    model_config = ModelConfig(
        d_model=256,
        n_head=8,
        n_enc_layers=4,
        n_dec_layers=4,
        dim_ff=1024,
        dropout=0.1,
    )

    train_config = TrainConfig(
        max_train_steps=20_000,
        lr=5e-4,
        valid_every=500,
        max_gen_len=30,
        show_valid_samples=5,
        num_epochs=100,
        save_best_path="best_model.pt",
    )

    wandb.init(
        project="inthon-arithmetic-transformer",
        name="transformer-v1",
        config={**train_config.__dict__, **model_config.__dict__},
    )

    model = TransformerSeq2Seq(
        in_vocab=input_tokenizer.vocab_size,
        out_vocab=output_tokenizer.vocab_size,
        **model_config.__dict__,
    )

    wandb.watch(model, log="all", log_freq=200)

    train_loop(
        model=model,
        dataloader=train_dataloader,
        input_tokenizer=input_tokenizer,
        output_tokenizer=output_tokenizer,
        device=device,
        val_dataloader=val_dataloader,
        train_config=train_config,
        model_config=model_config,
        tokenizer_config=tokenizer_config,
    )

    torch.save(model.state_dict(), "model.pt")
    print("Saved model.pt")

    wandb.finish()


if __name__ == "__main__":
    main()