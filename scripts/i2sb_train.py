# scripts/i2sb_train.py
# -*- coding: utf-8 -*-
"""
I2SB local multi trainer entry.

新增（最小侵入）：
- --init_from: 仅加载模型权重（state_dict），不恢复 optimizer/scheduler/scaler/EMA/best/global_step
- --init_strict: init_from 是否 strict load（默认 true）
- --start_epoch: 伪续训时从多少 epoch 记（默认 151）
  - 注意：trainer 内部依然会从 start_epoch 开始循环 epoch。
  - 如果从 151 开始继续训练” 那 epochs 仍然是 cfg.train.epochs（300）；
    如果只是想 日志从151记但只再训练150个epoch，那 epochs 设为 300（151~300 共150个）。
"""

import argparse
import yaml
import os
import inspect
from typing import Any, List, Optional, Dict, Union

import torch

from ctprojfix.data.dataset import make_dataloader
from ctprojfix.models.i2sb_unet import I2SBUNet


def load_cfg(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _str2bool(x: Optional[str]) -> Optional[bool]:
    if x is None:
        return None
    s = str(x).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise ValueError(f"Cannot parse bool from: {x}")


def _resolve_device(cfg_train: dict, cli_device: Optional[str], cli_gpu: Optional[Union[str, int]]) -> torch.device:
    """
    优先级：
      1) --device
      2) --gpu
      3) cfg.train.device
      4) auto
    """
    if cli_device:
        d = str(cli_device).strip().lower()
        if d == "cpu":
            return torch.device("cpu")
        if d.startswith("cuda") and torch.cuda.is_available():
            return torch.device(d)
        print(f"[WARN] --device={d} 不可用，退回 CPU")
        return torch.device("cpu")

    if cli_gpu is not None:
        try:
            idx = int(cli_gpu)
            if torch.cuda.is_available():
                return torch.device(f"cuda:{idx}")
        except Exception:
            pass
        print(f"[WARN] --gpu={cli_gpu} 不可用，退回 CPU")
        return torch.device("cpu")

    d = str(cfg_train.get("device", "") or "").strip().lower()
    if d:
        if d == "cpu":
            return torch.device("cpu")
        if d.startswith("cuda") and torch.cuda.is_available():
            return torch.device(d)
        if d.startswith("cuda"):
            print(f"[WARN] cfg.train.device={d} 要求 CUDA，但本机不可用，退回 CPU。")
            return torch.device("cpu")

    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def _maybe_wrap_dataparallel(model: torch.nn.Module, cfg_train: dict) -> torch.nn.Module:
    """
    默认关闭；只有 cfg.train.data_parallel=true 才会启用。
    """
    use_dp = bool(cfg_train.get("data_parallel", False))
    gpu_ids: List[int] = cfg_train.get("gpu_ids", [])
    if not use_dp:
        return model
    if not torch.cuda.is_available():
        print("[WARN] data_parallel=True 但 CUDA 不可用，忽略。")
        return model
    if not gpu_ids:
        gpu_ids = list(range(torch.cuda.device_count()))
    if len(gpu_ids) <= 1:
        return model
    print(f"[DP] Using DataParallel on GPU ids: {gpu_ids}")
    return torch.nn.DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])


def _instantiate_with_filtered_kwargs(cls, **kwargs):
    """
    只把 __init__ 支持的参数传进去（防止 cfg/CLI 多写 key 直接炸）。
    """
    sig = inspect.signature(cls.__init__)
    accepted = set(sig.parameters.keys())
    accepted.discard("self")

    used, ignored = {}, {}
    for k, v in kwargs.items():
        if k in accepted:
            used[k] = v
        else:
            ignored[k] = v

    if ignored:
        print(f"[INFO] {cls.__name__} ignores keys: {sorted(list(ignored.keys()))}")
    return cls(**used)


def _put_if_not_none(d: dict, k: str, v):
    if v is not None:
        d[k] = v


def _is_dataparallel(model: torch.nn.Module) -> bool:
    return isinstance(model, torch.nn.DataParallel)


def _strip_module_prefix_if_needed(state: Dict[str, Any], model_is_dp: bool) -> Dict[str, Any]:
    """
    仅当「目标模型不是 DP」且「state_dict 带 module.」时，才 strip。
    - model_is_dp=True：保持 module. 前缀（否则会导致 strict=True 失配）
    - model_is_dp=False：去掉 module.，便于加载到非 DP 模型
    """
    if not isinstance(state, dict) or not state:
        return state
    has_module = any(k.startswith("module.") for k in state.keys())
    if has_module and (not model_is_dp):
        return {k[len("module."):]: v for k, v in state.items()}
    return state


def _add_module_prefix_if_needed(state: Dict[str, Any], model_is_dp: bool) -> Dict[str, Any]:
    """
    仅当「目标模型是 DP」且「state_dict 不带 module.」时，加 module. 前缀。
    这样 DP/非DP 都能互相 warm start。
    """
    if not isinstance(state, dict) or not state:
        return state
    has_module = any(k.startswith("module.") for k in state.keys())
    if (not has_module) and model_is_dp:
        return {("module." + k): v for k, v in state.items()}
    return state


def _load_weights_only(model: torch.nn.Module, ckpt_path: str, device: torch.device, strict: bool = True):
    """
    仅加载模型权重（state_dict），不恢复 optimizer/scheduler/scaler/EMA/best/global_step。
    兼容：
      - ckpt 是 dict 且包含 'state_dict'
      - ckpt 直接就是 state_dict
      - DP/non-DP 前缀互转（module.）
    """
    if (not ckpt_path) or (not os.path.isfile(ckpt_path)):
        raise FileNotFoundError(f"[INIT] init_from ckpt not found: {ckpt_path}")

    ck = torch.load(ckpt_path, map_location=device)
    sd = ck.get("state_dict", ck)

    model_is_dp = _is_dataparallel(model)
    sd = _strip_module_prefix_if_needed(sd, model_is_dp=model_is_dp)
    sd = _add_module_prefix_if_needed(sd, model_is_dp=model_is_dp)

    missing, unexpected = model.load_state_dict(sd, strict=strict)
    print(f"[INIT] loaded weights-only from: {ckpt_path} | strict={strict} | model_dp={model_is_dp}")

    if (not strict) and (missing or unexpected):
        print(f"[INIT] missing_keys={len(missing)} unexpected_keys={len(unexpected)}")
        if len(missing) > 0:
            print(f"[INIT]  missing (first 10): {missing[:10]}")
        if len(unexpected) > 0:
            print(f"[INIT]  unexpected (first 10): {unexpected[:10]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="configs/train_i2sb_local_multi.yaml")
    ap.add_argument("--device", type=str, default=None, help='如 "cuda:2" / "cpu"（优先级最高）')
    ap.add_argument("--gpu", type=str, default=None, help="等价于 --device cuda:{idx}")

    # 续训与评估频率
    ap.add_argument("--resume_from", type=str, default=None, help="指定 ckpt 路径继续训练（可选）")
    ap.add_argument("--resume", type=str, default=None, help='auto/none/last/best（可选，覆盖 cfg.train.resume）')
    ap.add_argument("--strict_load", type=str, default=None, help='true/false（覆盖 cfg.train.strict_load）')

    ap.add_argument("--eval_every", type=int, default=None, help="每 N 个 epoch 才跑一次 val（可选）")
    ap.add_argument("--val_max_batches", type=int, default=None, help="val 只跑前 N 个 batch（0=全量）（可选）")

    # 预览图频率：对应 trainer.dump_preview_every
    ap.add_argument("--preview_every", type=int, default=None, help="每 N 个 epoch 导出一次 preview 图（0=关闭）（可选）")

    # sampler (val/preview)
    ap.add_argument("--val_infer", type=str, default=None, help='one_step/sample（覆盖 cfg.train.val_infer）')
    ap.add_argument("--sample_steps", type=int, default=None, help="采样步数（覆盖 cfg.train.sample_steps）")
    ap.add_argument("--sample_stochastic", type=str, default=None, help="true/false（覆盖 cfg.train.sample_stochastic）")
    ap.add_argument("--sample_clamp_known", type=str, default=None, help="true/false（覆盖 cfg.train.sample_clamp_known）")

    # metric direction
    ap.add_argument("--maximize_metric", type=str, default=None,
                    help="true/false（覆盖 cfg.train.maximize_metric；不填则交给 trainer 自动推断）")

    # ---------------- NEW: warm start (weights-only) ----------------
    ap.add_argument("--init_from", type=str, default=None,
                    help="仅加载模型权重做伪续训/迁移训练（不恢复 optimizer/scheduler/scaler/EMA/best/global_step）")
    ap.add_argument("--init_strict", type=str, default="true",
                    help="true/false：init_from 加载是否严格匹配（默认 true）")
    ap.add_argument("--start_epoch", type=int, default=151,
                    help="伪续训时从多少 epoch 开始记（默认 151）。会覆盖 trainer 的 start_epoch（但不恢复旧训练状态）")

    args = ap.parse_args()

    cfg = load_cfg(args.cfg)
    tr = cfg.get("train", {}) or {}
    model_cfg = cfg.get("model", {}) or {}
    name_lower = str(model_cfg.get("name", "")).lower().strip()

    # 只考虑 i2sb_local_multi
    if name_lower != "i2sb_local_multi":
        raise RuntimeError(
            f"[i2sb_train.py] 只支持 model.name=i2sb_local_multi，但给的是: {model_cfg.get('name')}"
        )

    # ---------------- unified init_from / init_strict / start_epoch (CLI > CFG) ----------------
    init_from = args.init_from
    if init_from is None or str(init_from).strip() == "":
        init_from = tr.get("init_from", None)

    init_strict = _str2bool(args.init_strict)
    if init_strict is None:
        init_strict = _str2bool(tr.get("init_strict", "true"))
    if init_strict is None:
        init_strict = True

    start_epoch = args.start_epoch
    # 若 CLI 没显式改（仍是默认 151），允许 cfg 覆盖
    if ("start_epoch" in tr) and (args.start_epoch == 151):
        try:
            start_epoch = int(tr.get("start_epoch", 151))
        except Exception:
            start_epoch = 151

    use_init = (init_from is not None) and (str(init_from).strip() != "")

    # device
    device = _resolve_device(tr, args.device, args.gpu)
    if device.type == "cuda":
        try:
            torch.cuda.set_device(device.index if device.index is not None else 0)
        except Exception as e:
            print(f"[WARN] torch.cuda.set_device 失败：{e}")

    print(f"[DEVICE] using device = {device}")
    if device.type == "cuda":
        try:
            print(f"[DEVICE] GPU name = {torch.cuda.get_device_name(device)}")
            print(f"[DEVICE] CUDA_VISIBLE_DEVICES = {os.environ.get('CUDA_VISIBLE_DEVICES', '(unset)')}")
        except Exception:
            pass

    # data
    if "data" not in cfg:
        raise RuntimeError("[CFG] missing top-level key: data")
    loaders = make_dataloader(cfg["data"])
    if isinstance(loaders, dict):
        train_loader = loaders.get("train")
        val_loader = loaders.get("val")
    else:
        train_loader, val_loader = loaders, None

    if train_loader is None:
        raise RuntimeError("[DATA] train_loader is None (check split/train_list).")

    try:
        n_tr = len(train_loader.dataset)
        n_va = len(val_loader.dataset) if val_loader is not None else 0
        print(f"[DATA] sizes -> train:{n_tr}  val:{n_va}")
    except Exception:
        pass

    # model (I2SBUNet) —— 白名单过滤，避免 emb_dim 参数名不匹配
    mp = model_cfg.get("params", {}) or {}
    model_kwargs = dict(
        in_ch=int(mp.get("in_ch", 5)),   # multi: [xt, x1, mask, angle, t]
        base=int(mp.get("base", 64)),
        depth=int(mp.get("depth", 4)),
        dropout=float(mp.get("dropout", 0.0)),
    )
    # 如果 cfg 里显式写了 emb_dim，才尝试传
    if "emb_dim" in mp and mp["emb_dim"] is not None:
        model_kwargs["emb_dim"] = int(mp["emb_dim"])

    model = _instantiate_with_filtered_kwargs(I2SBUNet, **model_kwargs).to(device)

    # 先决定是否 DP，再 wrap；weights-only load 在 wrap 后进行（支持 DP/nonDP key）
    model = _maybe_wrap_dataparallel(model, tr)

    # ---------------- weights-only warm start ----------------
    if use_init:
        print(f"[INIT] enabled: init_from={init_from} init_strict={init_strict} start_epoch={start_epoch}")
        _load_weights_only(model, str(init_from), device, strict=bool(init_strict))

    # trainer
    from ctprojfix.trainers.i2sb_local_multi import I2SBLocalTrainer as TrainerCls
    print("[INFO] Loading I2SBLocalMultiTrainer (random-t bridge, multi-step val/preview)...")

    # cfg + CLI override
    resume_from = args.resume_from if args.resume_from is not None else tr.get("resume_from", None)
    resume = args.resume if args.resume is not None else tr.get("resume", "auto")

    # 若启用 init_from：强制禁用 trainer resume（避免恢复 optimizer/scheduler/scaler/EMA/best/global_step）
    if use_init:
        resume = "none"
        resume_from = None

    strict_cli = _str2bool(args.strict_load)
    strict_load = strict_cli if strict_cli is not None else bool(tr.get("strict_load", True))

    eval_every = args.eval_every if args.eval_every is not None else int(tr.get("eval_every", 1))
    val_max_batches = args.val_max_batches if args.val_max_batches is not None else int(tr.get("val_max_batches", 0))

    preview_every = args.preview_every if args.preview_every is not None else int(tr.get("dump_preview_every", 0))

    # sampler overrides
    val_infer = args.val_infer if args.val_infer is not None else tr.get("val_infer", None)
    sample_steps = args.sample_steps if args.sample_steps is not None else tr.get("sample_steps", None)

    stoch_cli = _str2bool(args.sample_stochastic)
    sample_stochastic = stoch_cli if stoch_cli is not None else tr.get("sample_stochastic", None)

    clamp_cli = _str2bool(args.sample_clamp_known)
    sample_clamp_known = clamp_cli if clamp_cli is not None else tr.get("sample_clamp_known", None)

    # maximize_metric override: allow None -> let trainer infer
    max_cli = _str2bool(args.maximize_metric)
    maximize_metric = max_cli if max_cli is not None else tr.get("maximize_metric", None)

    # ema_decay robust parse
    if tr.get("ema_decay", None) is not None:
        ema_decay = float(tr.get("ema_decay"))
    else:
        # 兼容有人写 train.ema=0.999（老风格）
        ema_raw = tr.get("ema", None)
        ema_decay = float(ema_raw) if isinstance(ema_raw, (int, float)) else 0.999

    # 打印最终生效配置检查
    print(
        f"[RUN CFG] eval_every={eval_every}  val_max_batches={val_max_batches}  preview_every={preview_every}  "
        f"resume={resume}  resume_from={resume_from}  strict_load={strict_load}  "
        f"val_infer={val_infer if val_infer is not None else '(trainer default)'}  "
        f"sample_steps={sample_steps if sample_steps is not None else '(trainer default)'}  "
        f"sample_stochastic={sample_stochastic if sample_stochastic is not None else '(trainer default)'}  "
        f"sample_clamp_known={sample_clamp_known if sample_clamp_known is not None else '(trainer default)'}  "
        f"val_metric={tr.get('val_metric', 'loss')}  maximize_metric={maximize_metric if maximize_metric is not None else '(trainer auto)'}"
    )
    if use_init:
        print(f"[RUN CFG] weights-only init_from enabled -> force resume=none | start_epoch={start_epoch}")

    trainer_kwargs = dict(
        device=str(device),
        lr=float(tr.get("lr", 3e-4)),
        epochs=int(tr.get("epochs", 150)),
        sigma_T=float(tr.get("sigma_T", 1.0)),
        t0=float(tr.get("t0", 1e-4)),
        ema_decay=ema_decay,

        ckpt_dir=tr.get("ckpt_dir", "checkpoints/i2sb_local_multi"),
        ckpt_prefix=tr.get("ckpt_prefix", "i2sb_local_multi"),
        log_dir=tr.get("log_dir", "logs/i2sb_local_multi"),

        save_every=int(tr.get("save_every", 1)),
        max_keep=int(tr.get("max_keep", 5)),
        log_interval=int(tr.get("log_interval", 100)),

        # preview
        dump_preview_every=int(preview_every),

        # data cond (保持跟 data.add_angle_channel 一致)
        cond_has_angle=bool(cfg.get("data", {}).get("add_angle_channel", False)),

        # metric/sched
        val_metric=str(tr.get("val_metric", "loss")),
        maximize_metric=maximize_metric,   # 允许 None：交给 trainer 自动推断
        sched=tr.get("sched", None),

        # eval throttling
        eval_every=int(eval_every),
        val_max_batches=int(val_max_batches),

        # resume（若 init_from 启用，这里已被强制 none/None）
        resume_from=resume_from,
        resume=str(resume),
        strict_load=bool(strict_load),
    )

    # ---- optional trainer args: only pass if present (avoid float(None) crash) ----
    _put_if_not_none(trainer_kwargs, "w_valid", tr.get("w_valid", None))
    _put_if_not_none(trainer_kwargs, "w_missing", tr.get("w_missing", None))

    _put_if_not_none(trainer_kwargs, "use_percep", tr.get("use_percep", None))
    _put_if_not_none(trainer_kwargs, "w_percep", tr.get("w_percep", None))
    _put_if_not_none(trainer_kwargs, "percep_region", tr.get("percep_region", None))
    _put_if_not_none(trainer_kwargs, "percep_max_hw", tr.get("percep_max_hw", None))
    _put_if_not_none(trainer_kwargs, "percep_use_pretrained", tr.get("percep_use_pretrained", None))
    _put_if_not_none(trainer_kwargs, "percep_layers", tr.get("percep_layers", None))

    # sampler args (val/preview)
    _put_if_not_none(trainer_kwargs, "val_infer", val_infer)
    _put_if_not_none(trainer_kwargs, "sample_steps", sample_steps)
    _put_if_not_none(trainer_kwargs, "sample_stochastic", sample_stochastic)
    _put_if_not_none(trainer_kwargs, "sample_clamp_known", sample_clamp_known)

    trainer = _instantiate_with_filtered_kwargs(TrainerCls, **trainer_kwargs)

    # ---------------- force start_epoch when using init_from ----------------
    if use_init:
        # 只影响 epoch 计数与循环起点；不会恢复旧的 optimizer/scheduler 等状态
        try:
            trainer.start_epoch = int(start_epoch)
            print(f"[INIT] set trainer.start_epoch = {trainer.start_epoch}")
        except Exception as e:
            print(f"[WARN] set trainer.start_epoch failed: {e}")

    # train
    trainer.fit(model, train_loader, val_loader)


if __name__ == "__main__":
    main()
