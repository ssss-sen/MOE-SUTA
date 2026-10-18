# -*- coding: utf-8 -*-
"""
md_eval.py -- 在任意 task（md_easy / md_hard / LS_XX_5 ...）上跑一个策略，并额外输出 per-domain WER。
放在仓库根目录运行（和 run_benchmark.py 同级）。

用法示例：
  python md_eval.py -s moe_suta -t md_easy \
      --config config/system/moe-suta-lb001.yaml \
      --strategy_config config/strategy/5step.yaml \
      -n lb001 --set lb_coef=0.01 --repeat 1

  python md_eval.py -s dsuta -t md_hard \
      --strategy_config config/strategy/dsuta.yaml config/strategy/5step.yaml -n mdhard-dsuta

设计说明：
  * 完全复用仓库自己的 task 定义（src/tasks/load.py），**不需要外部数据脚本**；
  * 用一层 Dataset 包装器记录每个样本的 tid，从而在跑完后算出 per-domain WER
    （仓库 run_benchmark.py 只报总 WER，看不到"哪个域崩了"）；
  * 不改动任何策略代码，只调用 strategy.run(task)。
"""
import argparse
import gc
import json
import os
import sys
import time

import yaml

from src.utils.tool import seed_everything
from src.strategies.load import get_strategy_cls
from src.tasks.load import get_task


class TidTrace:
    """包装 task，记录每个被读取样本的 (tid, wav 长度, id)，用于事后对齐 results。"""

    def __init__(self, ds):
        self.ds = ds
        self.task_boundaries = getattr(ds, "task_boundaries", []) or []

        self.trace = []

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        s = self.ds[idx]
        wav = s.get("wav")
        self.trace.append({
            "tid": s.get("tid", -1),
            "n": len(wav) if wav is not None else 0,
            "id": s.get("id", str(idx)),
        })
        return s


def build_config(args):
    system_config = yaml.load(open(args.config, "r"), Loader=yaml.FullLoader)
    strategy_config = {}
    for path in args.strategy_config:
        strategy_config.update(yaml.load(open(path, "r"), Loader=yaml.FullLoader))
    for kv in (args.set or []):
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        system_config[k] = v
    for kv in (args.set_moetta or []):
        k, v = kv.split("=", 1)
        try:
            v = yaml.safe_load(v)
        except Exception:
            pass
        system_config.setdefault("moetta", {})[k] = v
    return {
        "strategy_name": args.strategy_name,
        "task_name": args.task_name,
        "system_config": system_config,
        "strategy_config": strategy_config,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-s", "--strategy_name", required=True)
    p.add_argument("-t", "--task_name", required=True)
    p.add_argument("-n", "--exp_name", default="md_eval")
    p.add_argument("--config", default="config/system/suta.yaml")
    p.add_argument("--strategy_config", nargs="+", default=["config/strategy/10step.yaml"])
    p.add_argument("--set", nargs="*", default=[], help="覆盖 system_config 顶层键，如 lb_coef=0.01")
    p.add_argument("--set-moetta", nargs="*", default=[], help="覆盖 moetta.* 键，如 num_expert=1")
    p.add_argument("--seed", type=int, default=666)
    p.add_argument("--repeat", type=int, default=1, help="重复跑几次（同一 seed，用于估计 run-to-run 波动）")
    p.add_argument("--limit", type=int, default=0, help="只跑前 N 条（冒烟测试用）")
    args = p.parse_args()

    config = build_config(args)
    exp_root = "results/md_eval/%s/%s/%s" % (args.strategy_name, args.exp_name, args.task_name)
    os.makedirs(exp_root, exist_ok=True)
    with open(os.path.join(exp_root, "config.yaml"), "w", encoding="utf-8") as f:
        yaml.dump(config, f, sort_keys=False)

    print("==== md_eval ====")
    print("strategy:", args.strategy_name, "| task:", args.task_name, "| exp:", args.exp_name)
    print("system_cfg(部分):", {k: config["system_config"].get(k)
                                for k in ("lr", "em_coef", "temp", "blank_coef", "nb_threshold", "lb_coef")})
    print("moetta:", {k: config["system_config"].get("moetta", {}).get(k)
                      for k in ("num_expert", "topk", "randomness", "lb_coeff", "dynamic_lb")})
    print("strategy_cfg:", config["strategy_config"])
    print("extra env: MOE_FORCE_RANDOM=%s" % os.environ.get("MOE_FORCE_RANDOM", "(unset)"))

    summary = []
    for rep in range(args.repeat):
        gc.collect()
        seed_everything(args.seed + rep)   # 每次 repeat 用不同 seed，才能反映真实 run-to-run 波动
        strategy = get_strategy_cls(args.strategy_name)(config)
        task = get_task(args.task_name)
        if getattr(args, "limit", 0) and args.limit > 0:
            from torch.utils.data import Subset
            task = Subset(task, list(range(min(args.limit, len(task)))))
        traced = TidTrace(task)
        keep = config["strategy_config"].get("max_length", 10 ** 12)

        t0 = time.time()
        results = strategy.run(traced)
        dt = time.time() - t0

        wers, nw = results["wers"], results["n_words"]
        num = sum(w * n for w, n in zip(wers, nw))
        den = sum(nw)
        total = 100.0 * num / den

        # 对齐 tid：策略会跳过 len(wav) > max_length 的样本，这里用同一规则重建
        kept = [tr for tr in traced.trace if tr["n"] <= keep]
        per_domain = {}
        if len(kept) == len(wers) and kept and all(tr["tid"] >= 0 for tr in kept):
            agg = {}
            for tr, w, n in zip(kept, wers, nw):
                a = agg.setdefault(tr["tid"], [0.0, 0])
                a[0] += w * n
                a[1] += n
            per_domain = {k: round(100.0 * v[0] / v[1], 2) for k, v in sorted(agg.items())}
        else:
            print("[warn] tid 对齐失败：kept=%d results=%d（task 可能没有 tid，或 max_length 规则不同）"
                  % (len(kept), len(wers)))

        n_skipped = len(traced.trace) - len(kept)
        print("\n[repeat %d] WER=%.2f%%  time=%.1fs  adapt=%s  skipped(too long)=%d"
              % (rep, total, dt, strategy.get_adapt_count(), n_skipped))
        if per_domain:
            print("[repeat %d] per-domain WER: %s" % (rep, per_domain), flush=True)
        summary.append({"repeat": rep, "wer": round(total, 2), "seconds": round(dt, 1),
                        "adapt": strategy.get_adapt_count(), "per_domain": per_domain})

        with open(os.path.join(exp_root, "results_%d.pkl" % rep), "wb") as f:
            import pickle
            pickle.dump(results, f)

    if args.repeat > 1:
        ws = [s["wer"] for s in summary]
        print("\n==== 汇总 (repeat=%d) ====" % args.repeat)
        print("WER per run:", ws)
        print("mean=%.2f  min=%.2f  max=%.2f  极差=%.2f" % (sum(ws) / len(ws), min(ws), max(ws),
                                                          max(ws) - min(ws)))
    with open(os.path.join(exp_root, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print("\nsaved to", exp_root)


if __name__ == "__main__":
    main()
