import numpy as np
import torch
import torch.nn as nn
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
from copy import deepcopy
import json

from ..utils.tool import batchify
from .loss import softmax_entropy, mcc_loss, div_loss
from .moe_model import MoETTA

def parse_layer_string(layer_str):
    if not layer_str:
        return []
    if isinstance(layer_str, int):
        return [layer_str]
    layer_str = str(layer_str)
    if '-' in layer_str:
        start, end = map(int, layer_str.split('-'))
        return list(range(start, end + 1))
    return [int(x) for x in layer_str.split(',') if x]


class SUTASystem(object):

    SAMPLE_RATE = 16000

    def __init__(self, config) -> None:
        self.config = config
        self.history = {}
        self.adapt_count = 0

        # load processor and model
        raw_processor_no_lm = Wav2Vec2Processor.from_pretrained(config["model_name"], sampling_rate=SUTASystem.SAMPLE_RATE)
        self.raw_processor_no_lm = raw_processor_no_lm
        if config.get("use_lm", False) or config["model_name"] == "patrickvonplaten/wav2vec2-base-960h-4-gram":
            from transformers import Wav2Vec2ProcessorWithLM
            ngram_decoder = Wav2Vec2ProcessorWithLM.from_pretrained("patrickvonplaten/wav2vec2-base-960h-4-gram").decoder
            self.processor = Wav2Vec2ProcessorWithLM(
                feature_extractor=raw_processor_no_lm.feature_extractor,
                tokenizer=raw_processor_no_lm.tokenizer,
                decoder=ngram_decoder
            )
        else:
            self.processor = raw_processor_no_lm
        
        if config["model_name"] == "facebook/data2vec-audio-base-960h":
            from transformers import Data2VecAudioForCTC
            self.model = Data2VecAudioForCTC.from_pretrained(config["model_name"], ctc_loss_reduction="mean")
        elif config["model_name"] == "facebook/hubert-large-ls960-ft":
            from transformers import HubertForCTC
            self.model = HubertForCTC.from_pretrained(config["model_name"], ctc_loss_reduction="mean")
        else:
            self.model = Wav2Vec2ForCTC.from_pretrained(config["model_name"], ctc_loss_reduction="mean")  # be careful that we need to use mean
        
        self.is_moe = config.get("use_moe", False)
        if self.is_moe:
            moe_cfg = config.get("moetta", {})
            disabled_layers = parse_layer_string(moe_cfg.get("disabled_layer", "0-0"))
            normal_layers = parse_layer_string(moe_cfg.get("normal_layer", ""))
            self.model = MoETTA(
                model=self.model,
                num_expert=moe_cfg.get("num_expert", 9),
                topk=moe_cfg.get("topk", 1),
                hidden_dim=moe_cfg.get("hidden_dim", 768),
                frames_per_sec=moe_cfg.get("frames_per_sec", 50),
                samplewise=moe_cfg.get("samplewise", True),
                dynamic_threshold=moe_cfg.get("dynamic_threshold", True),
                dynamic_lb=moe_cfg.get("dynamic_lb", True),
                lb_coeff=moe_cfg.get("lb_coeff", 0.2),
                weight_by_prob=moe_cfg.get("weight_by_prob", True),
                weight_by_entropy=moe_cfg.get("weight_by_entropy", True),
                e_margin_coeff=moe_cfg.get("e_margin_coeff", 0.4),
                randomness=moe_cfg.get("randomness", 0.0),
                activate_shared_expert=moe_cfg.get("activate_shared_expert", False),
                route_penalty=moe_cfg.get("route_penalty", 0.0),
                decay=moe_cfg.get("decay", 0.0),
                self_router=moe_cfg.get("self_router", True),
                grad_hook=moe_cfg.get("grad_hook", False),
                disabled_layer=disabled_layers,
                normal_layer=normal_layers,
                pass_through_coeff=moe_cfg.get("pass_through_coeff", True),
                global_router_idx=moe_cfg.get("global_router_idx", -1),
                lb_window=int(moe_cfg.get("lb_window", 1) or 1),  # [V1-lbwindow]
                batch_route=bool(moe_cfg.get("batch_route", False)),          # [BR]
                batch_route_momentum=float(moe_cfg.get("batch_route_momentum", 0.5)),  # [BR]
                route_granularity=str(moe_cfg.get("route_granularity", "native") or "native"),  # [RG]
                expert_schedule=str(moe_cfg.get("expert_schedule", "") or ""),                # [ES]
                device=config.get("device", "cuda")
            )
        
        self.model.train()  # huggingface default loads with eval mode
        self.model.cuda()

        # set up for tent
        self.optimizer, self.scheduler = setup_optimizer(
            self.build_optimized_model(),
            config["opt"], config["lr"], scheduler=config["scheduler"]
        )

        f = open('vocab.json')
        self.vocab = json.load(f)

        self.snapshot("init")

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    # ==================== [EXPERT_LR_PATCH] ====================
    # 把官方 MoETTA 的 lr 值施加在【专家 + router】上，其余保持本仓库实测最优的 lr。
    # 门控：环境变量 MOE_EXPERT_LR（默认不设 => 单 param_group，行为与历史结果逐位一致）
    def _apply_expert_lr(self, params):
        import os
        exp_lr = os.environ.get("MOE_EXPERT_LR", "").strip()
        base_lr = self.config.get("lr", None)
        if not exp_lr:
            return params
        if base_lr is None:
            print("[EXPERT_LR] config most lr key => fallback single group", flush=True)
            return params
        exp_lr, base_lr = float(exp_lr), float(base_lr)
        ex_ids = set()
        layers = getattr(self.model, "moe_layer", None) or []
        for m in layers:
            for nm in ("experts_weight", "experts_bias"):
                _p = getattr(m, nm, None)
                if isinstance(_p, torch.nn.Parameter):
                    ex_ids.add(id(_p))
            _r = getattr(m, "router", None)
            if _r is not None:
                for _p in _r.parameters():
                    ex_ids.add(id(_p))
        grp_ex = [p for p in params if id(p) in ex_ids]
        grp_rest = [p for p in params if id(p) not in ex_ids]
        if not grp_ex:
            print("[EXPERT_LR] WARN no experts/router matched => fallback single group", flush=True)
            return params
        _m = lambda g: sum(p.numel() for p in g) / 1e6
        print("[EXPERT_LR] expert/router lr=%.3e (rest lr=%.3e, =%.0fx) | grp=%d tensors/%.3fM | rest=%d tensors/%.3fM"
              % (exp_lr, base_lr, (exp_lr / base_lr) if base_lr else -1.0,
                 len(grp_ex), _m(grp_ex), len(grp_rest), _m(grp_rest)), flush=True)
        return [{"params": grp_rest}, {"params": grp_ex, "lr": exp_lr}]

    def build_optimized_model(self):
        if self.is_moe:
            params = self.model.get_params()
            self.opt_param_names = ["moe_params"] * len(params)
            for p in params:
                p.requires_grad = True
            print("MoE Optimizable Params: ", sum(p.numel() for p in params))
            # [EXPERT_LR_PATCH]
            return self._apply_expert_lr(params)
        else:
            self.model.requires_grad_(False)
            params, self.opt_param_names = self.collect_params()
            # print(param_names[:10])
            for p in params:
                p.requires_grad = True
            print("Optimizable: ", sum(p.numel() for p in self.model.parameters() if p.requires_grad))
            return params

    def _wav_to_model_input(self, wavs):
        # Due to wav2vec2-base special design, no attention mask is returned.
        # Wav2Vec2Processor's default argument for return_attention_mask will be False here.
        # However, it will be True in other speech models!
        inputs = self.processor(
            audio=wavs,
            sampling_rate=SUTASystem.SAMPLE_RATE,
            return_tensors="pt",
            padding="longest",
        )
        return inputs.to(device=self.model.device)
    
    def _text_to_model_input(self, texts):
        # target texts need to match wav2vec2's format to make sure correct tokenization
        texts_new = []
        for x in texts:
            x = x.upper()
            x_new = ""
            for s in x:
                if s in self.vocab or s == ' ':
                    x_new += s
            texts_new.append(x_new)

        labels = self.processor(
            text=texts_new,
            return_tensors="pt",
            padding="longest",
        )
        labels = labels.input_ids.masked_fill(labels.attention_mask.ne(1), -100)
        return labels.to(device=self.model.device)

    def reset_adapt_counter(self):
        self.adapt_count = 0
    
    def l2_loss(self):
        l2_loss = 0.0
        orig_state_dict = self.history["init"][0]
        _cache = getattr(self, "_l2_init_dev", None)
        if _cache is None:
            _cache = self._l2_init_dev = {}
        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            _ref = _cache.get(name)
            if _ref is None or _ref.device != param.device:
                _cand = orig_state_dict.get(name)
                if _cand is None:
                    continue
                _ref = _cand.to(param.device)
                _cache[name] = _ref
            l2_loss += torch.sum((param - _ref) ** 2)
        return l2_loss

    def suta_adapt(self, wavs, record={}):
        """
        Single gradient step on batch of data.
        Measure entropy of the model prediction, take gradients, and update params.
        the index of <pad> in vocab is 0
        Due to wav2vec2-base special design, attention mask is always none, so ctc input length is always the
        full length, model should learn to output id=0 on the padded part of wav.
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        # print(type(inputs))  # inputs belongs to a custom dict class defined in transformers, not tensor
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs)
        predicted_ids = torch.argmax(outputs.logits, dim=-1)

        loss = 0
        if self.config["em_coef"] > 0:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            x = softmax_entropy(outputs.logits / self.config["temp"])
            if self.config["non_blank"]:
                x = x[non_blank]
            if len(x) > 0:
                e_loss = x.mean(0).mean()
            else:
                e_loss = torch.tensor(0, device=self.model.device)
                record["collapse"] = True
            loss += e_loss * self.config["em_coef"]
            record["e_loss"] = e_loss.item()
        
        if 1 - self.config["em_coef"] > 0: 
            c_loss = mcc_loss(outputs.logits / self.config["temp"], self.config["reweight"])
            loss += c_loss * (1 - self.config["em_coef"])
            record["c_loss"] = c_loss.item()

        if self.config["div_coef"] > 0: 
            d_loss = div_loss(outputs.logits, self.config["non_blank"]) 
            loss += d_loss * self.config["div_coef"]
            record["d_loss"] = d_loss.item()
        
        if self.config["l2_coef"] > 0: 
            l2_loss = self.l2_loss()
            loss += l2_loss * self.config["l2_coef"]
            record["l2_loss"] = l2_loss.item()

        record["total_loss"] = loss.item()

        self.model.zero_grad()
        loss.backward()
        # print(e_loss.item(), c_loss.item(), l2_loss.item())
        # print(predicted_ids)
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()

    
    def suta_adapt_loss_only(self, wavs, record={}):
        """
        suta_adapt without gradient control so that we can use gradient accumulation
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        # print(type(inputs))  # inputs belongs to a custom dict class defined in transformers, not tensor
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs)
        predicted_ids = torch.argmax(outputs.logits, dim=-1)

        loss = 0
        if self.config["em_coef"] > 0:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            x = softmax_entropy(outputs.logits / self.config["temp"])
            if self.config["non_blank"]:
                x = x[non_blank]
            if len(x) > 0:
                e_loss = x.mean(0).mean()
            else:
                e_loss = torch.tensor(0, device=self.model.device)
                record["collapse"] = True
            loss += e_loss * self.config["em_coef"]
            record["e_loss"] = e_loss.item()
        
        if 1 - self.config["em_coef"] > 0: 
            c_loss = mcc_loss(outputs.logits / self.config["temp"], self.config["reweight"])
            loss += c_loss * (1 - self.config["em_coef"])
            record["c_loss"] = c_loss.item()

        if self.config["div_coef"] > 0: 
            d_loss = div_loss(outputs.logits, self.config["non_blank"]) 
            loss += d_loss * self.config["div_coef"]
            record["d_loss"] = d_loss.item()
        
        if self.config["l2_coef"] > 0: 
            l2_loss = self.l2_loss()
            loss += l2_loss * self.config["l2_coef"]
            record["l2_loss"] = l2_loss.item()

        record["total_loss"] = loss.item()

        return loss

    def suta_adapt_auto(self, wavs, batch_size=-1, record={}) -> None:
        """ suta_adapt auto split to smaller batch """
        self.adapt_count += 1
        if batch_size == -1:
            batch_size = len(wavs)
        self.model.zero_grad()
        denom_scale = len(wavs) // batch_size
        assert denom_scale > 0
        for wavs in batchify(wavs, batch_size=batch_size):
            loss = self.suta_adapt_loss_only(wavs, record=record)
            self.adapt_count -= 1  # avoid repeat count
            loss = loss / denom_scale
            loss.backward()
    
        self.optimizer.step()
        self.model.zero_grad()

    def ctc_adapt(self, wavs, texts, record={}):
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        labels = self._text_to_model_input(texts)
        if labels.shape[1] == 0:  # empty string exception, e.g. PL collapse
            labels = torch.zeros((len(labels), 1), device=labels.device)
            record["collapse"] = True
        inputs["labels"] = labels

        outputs = self.model(**inputs)
        loss = outputs.loss
        record["ctc_loss"] = loss.item()
        record["total_loss"] = loss.item()

        self.model.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
        self.model.zero_grad()

    def ctc_adapt_loss_only(self, wavs, texts, record={}):
        """
        ctc_adapt without gradient control so that we can use gradient accumulation
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        labels = self._text_to_model_input(texts)
        if labels.shape[1] == 0:  # empty string exception, e.g. PL collapse
            labels = torch.zeros((len(labels), 1), device=labels.device)
            record["collapse"] = True
        inputs["labels"] = labels

        outputs = self.model(**inputs)
        loss = outputs.loss
        record["ctc_loss"] = loss.item()
        record["total_loss"] = loss.item()

        return loss

    def ctc_adapt_auto(self, wavs, texts, batch_size=-1, record={}) -> None:
        """ ctc_adapt auto split to smaller batch """
        self.adapt_count += 1
        if batch_size == -1:
            batch_size = len(wavs)
        self.model.zero_grad()
        denom_scale = len(wavs) // batch_size
        assert denom_scale > 0
        for wavs, texts in zip(batchify(wavs, batch_size=batch_size), batchify(texts, batch_size=batch_size)):
            loss = self.ctc_adapt_loss_only(wavs, texts, record=record)
            self.adapt_count -= 1  # avoid repeat count
            loss = loss / denom_scale
            loss.backward()
    
        self.optimizer.step()
        self.model.zero_grad()

    def moe_suta_adapt(self, wavs, record={}):
        """
        使用 MOE 方式进行 TTA 推理和适应。
        利用 MoETTA 模型内部计算的 valid_mask，屏蔽高熵样本，使其不参与 MCC 和 DIV 等额外损失的参数更新。
        """
        self.adapt_count += 1

        inputs = self._wav_to_model_input(wavs)
        inputs = inputs.to(device=self.model.device)
        
        # 前向传播：在 MoETTA 内部会自动计算 valid_mask 和 moe_loss
        # （eval 下默认不产统计量；适应时显式打开，模型仍保持 eval/dropout=off）
        prev_flag = getattr(self.model, "collect_stats", False)
        self.model.collect_stats = True
        try:
            # 温度对齐：MoETTA 内部熵/valid_mask/coeff 与下方 MCC 使用同一 temp（默认2.5）
            self.model.current_temp = float(self.config.get("temp", 2.5))
            outputs = self.model(**inputs)
        finally:
            self.model.collect_stats = prev_flag
        predicted_ids = torch.argmax(outputs.logits, dim=-1)

        # EXP-F：完全采用 SUTA 原版损失（无权熵 on 非blank、MCC 全帧、无 LB、无置信度掩码）
        loss = 0

        if self.config["em_coef"] > 0:
            non_blank = torch.where(predicted_ids != 0, 1, 0).bool()
            x = softmax_entropy(outputs.logits / self.config["temp"])
            if self.config["non_blank"]:
                x = x[non_blank]
            if len(x) > 0:
                e_loss = x.mean(0).mean()
            else:
                e_loss = torch.tensor(0, device=self.model.device)
                record["collapse"] = True
            loss += e_loss * self.config["em_coef"]
            record["e_loss"] = e_loss.item()

        if 1 - self.config["em_coef"] > 0:
            c_loss = mcc_loss(outputs.logits / self.config["temp"], self.config["reweight"])
            loss += c_loss * (1 - self.config["em_coef"])
            record["c_loss"] = c_loss.item()

        if self.config["div_coef"] > 0:
            d_loss = div_loss(outputs.logits, self.config["non_blank"])
            loss += d_loss * self.config["div_coef"]
            record["d_loss"] = d_loss.item()

        if self.config.get("l2_coef", 0.0) > 0:
            l2_loss = self.l2_loss()
            loss += l2_loss * self.config.get("l2_coef", 0.0)
            record["l2_loss"] = l2_loss.item()

        # [BLANK_FOR_BASELINE] 给基线策略补上与 MoE 路径一致的"反 blank"项。
        #   默认 blank_coef 不存在 ⇒ 0 ⇒ 不加（历史结果完全不变）；显式 --set blank_coef=0.5 才启用。
        _bc = float(self.config.get("blank_coef", 0.0) or 0.0)
        if _bc > 0:
            nb_ratio = (predicted_ids != 0).float().mean()
            blank_prob = outputs.logits.softmax(-1)[..., 0].mean()
            record["nb_ratio"] = nb_ratio.item()
            if nb_ratio < self.config.get("nb_threshold", 0.2):
                loss = loss + _bc * blank_prob
                record["blank_penalty"] = blank_prob.item()

        # EXP-J: 反 blank 项（非blank帧比例过低时惩罚 blank 概率，抑制 CTC 空白坍缩）
        nb_ratio = (predicted_ids != 0).float().mean()
        blank_prob = outputs.logits.softmax(-1)[..., 0].mean()
        record["nb_ratio"] = nb_ratio.item()
        if nb_ratio < self.config.get("nb_threshold", 0.2):
            loss = loss + self.config.get("blank_coef", 0.1) * blank_prob
            record["blank_penalty"] = blank_prob.item()
        # [MOETTA_FILTER] 官方 MoETTA 的"动态阈值过滤 + 熵重加权"（Eq.8-10）
        #   用模型 forward 已算好的 valid_mask / current_coeff；env MOE_MOETTA_FILTER=1 才启用（默认关）
        #   实现：loss += (加权熵 − 原熵)·em_coef ⇒ 等价于把熵项替换成 Eq.9 的形式
        import os as _os
        if _os.environ.get("MOE_MOETTA_FILTER", "0") == "1" and self.config["em_coef"] > 0:
            _vm = getattr(self.model, "valid_mask", None)
            _cw = getattr(self.model, "current_coeff", None)
            if _vm is not None and _cw is not None:
                _H2 = softmax_entropy(outputs.logits / self.config["temp"])
                _m = _vm if not self.config["non_blank"] else (_vm & (predicted_ids != 0))
                _e_f = None
                if int(_m.sum()) > 0:
                    _e_f = (_H2[_m] * _cw[_m]).sum() / _m.sum()
                    loss = loss + (_e_f - e_loss.detach()) * self.config["em_coef"]
                    record["e_loss_filtered"] = float(_e_f.item())
                self._mf_n = getattr(self, "_mf_n", 0) + 1
                _nb = int((predicted_ids != 0).sum())
                valid_n = int(_vm.sum())
                if self._mf_n % 500 == 0:
                    print("[MF2] n=%d | S_t=%d | valid=%d | nonblank=%d | 帧=%d | 保留(全体)=%.1f%% 保留(nb)=%.1f%%"
                          % (self._mf_n, int(_m.sum()), valid_n, _nb, int(_m.numel()),
                             100.0*int(_m.sum())/max(1,int(_m.numel())), 100.0*int(_m.sum())/max(1,_nb)), flush=True)
                    print("[MF] n=%d | 可靠帧=%d/%d (%.1f%%) | 加权熵=%s | 原熵=%.4f"
                          % (self._mf_n, int(_m.sum()), int(_m.numel()),
                             100.0 * float(_m.sum()) / max(1, int(_m.numel())),
                             ("%.4f" % float(_e_f.item())) if _e_f is not None else "n/a",
                             float(e_loss.item())), flush=True)
        # [ENTROPY_WEIGHT] Q3 软加权：低置信度样本降低该步有效学习率（不丢样本）
        #   w=(H_ema/H)^gamma，再除以 w 的 EMA ⇒ 只做再分配、不改整体步长；clamp [0.2,2.0]
        #   门控：env MOE_ENT_W=1（默认关 ⇒ 与历史完全一致）；gamma = config.ent_w_gamma（默认 0.5）
        import os
        if os.environ.get("MOE_ENT_W", "0") == "1" and os.environ.get("MOE_ENTW_SCOPE", "both").strip().lower() != "slow":
            try:
                _H = float(e_loss.detach().item())
            except Exception:
                _H = 0.0
            if getattr(self, "_ent_H_ema", None) is None:
                self._ent_H_ema, self._ent_w_ema = max(_H, 1e-6), 1.0
            _gam = float(self.config.get("ent_w_gamma", 0.5))
            _mode = os.environ.get("MOE_ENTW_MODE", "soft").strip().lower()
            if _mode == "hard":
                _w = 0.0 if _H > float(self.config.get("ent_w_hard_k", 1.5)) * self._ent_H_ema else 1.0
            else:
                _w = (self._ent_H_ema / max(_H, 1e-6)) ** _gam
                _w = min(max(_w / max(self._ent_w_ema, 1e-6), 0.2), 2.0)
            loss = loss * _w
            self._ent_n = getattr(self, "_ent_n", 0) + 1
            if _w < 0.5:
                self._ent_drop = getattr(self, "_ent_drop", 0) + 1
            if self._ent_n % 500 == 0:
                print("[ENT_W] n=%d | 丢弃步数(w<0.5)=%d (%.1f%%) | H=%.4f H_ema=%.4f | mode=%s"
                      % (self._ent_n, getattr(self, "_ent_drop", 0),
                         100.0 * getattr(self, "_ent_drop", 0) / self._ent_n, _H, self._ent_H_ema, _mode), flush=True)
            self._ent_H_ema = 0.99 * self._ent_H_ema + 0.01 * _H
            self._ent_w_ema = 0.99 * self._ent_w_ema + 0.01 * _w
            record["ent_w"] = _w
            if not getattr(self, "_ent_w_logged", False):
                print("[ENT_W] 软加权启用 | gamma=%.2f | 首样本 H=%.4f w=%.4f" % (_gam, _H, _w), flush=True)
                self._ent_w_logged = True

        # EXP-K: 恢复官方 MoETTA 的 LB 项——给 router 一条“一阶直连 route_prob”的学习信号。
        # 说明：moe_model.forward 已算出 self.moe_lb_loss = lb_coeff * Σ_layers lb_loss
        #      （lb_coeff 取 config.moetta.lb_coeff=0.2 且 dynamic_lb 自适应），这里只决定“加不加、放大多少”。
        # 默认 lb_coef=0.0 → 与 EXP-F 行为完全一致（可复现历史结果）；>0 时启用。
        lb_coef = float(self.config.get("lb_coef", 0.0))
        moe_lb = getattr(self.model, "moe_lb_loss", None)
        if lb_coef > 0 and moe_lb is not None:
            loss = loss + lb_coef * moe_lb
            record["lb_loss"] = float(moe_lb.item())
            record["lb_coef"] = lb_coef
            record["lb_term"] = lb_coef * float(moe_lb.item())

        record["total_loss"] = loss.item()

        self.model.zero_grad()
        loss.backward()
        
        self.optimizer.step()
        if self.scheduler is not None: 
            self.scheduler.step()
            
        self.model.zero_grad()


    # inference
    @torch.no_grad()
    def inference(self, wavs, return_logits=False):
        inputs = self._wav_to_model_input(wavs)
        outputs = self.model(**inputs).logits
        predicted_ids = torch.argmax(outputs, dim=-1)
        transcription = self.raw_processor_no_lm.batch_decode(predicted_ids)
        
        if return_logits:
            logits = outputs.detach().cpu().numpy()
            return list(transcription), logits
        else:
            return list(transcription)
    
    @torch.no_grad()
    def beam_inference(self, wavs, n_best=1, text_only=True):
        """ Note that the underlying model should support beam search! """
        inputs = self._wav_to_model_input(wavs)
        logits = self.model(**inputs).logits
        # CAUTION:
        # See https://www.youtube.com/watch?v=mp7fHMTnK9A for definition of alpha and beta, and note that the defualt 
        # value of beta is not 0, which includes word length penalty and therefore not pure LM score
        if len(logits) == 1:  # no batch
            res = self.processor.decode(logits[0].cpu().numpy(), n_best=n_best, alpha=0.5, beta=0.0)
        else:
            res = self.processor.batch_decode(logits.cpu().numpy(), n_best=n_best, alpha=0.5, beta=0.0)
        if not text_only:
            return res
        transcription = res.text
        
        return list(transcription)
    
    # indicators
    @torch.no_grad()
    def calc_probability(self, wavs) -> float:
        assert len(wavs) == 1
        inputs = self._wav_to_model_input(wavs)  # inputs belongs to a custom dict class defined in transformers, not tensor
        inputs = inputs.to(device=self.model.device)
        outputs = self.model(**inputs).logits
        probability = outputs.log_softmax(dim=-1).sum().item()  # all experiments do not use length normalization to maintain simplicity

        return probability

    def calc_lm_score(self, text, normalized=False) -> float:
        self.processor.decoder.reset_params(  # CAUTION: need to reset to correct parameters or else will mismatch beam search scores!
            alpha=1.0, beta=0.0, unk_score_offset=None, lm_score_boundary=None
        )
        lm = self.processor.decoder._language_model
        raw_lm_score = 0.0
        start_state = lm.get_start_state()
        words = text.split(" ")
        n_word = len(words)
        for idx in range(n_word):
            score, start_state = lm.score(start_state, words[idx], is_last_word=False)
            # print("Word: ", words[idx], score)
            raw_lm_score += score
        
        # finalize
        score, _ = lm.score(start_state, "", is_last_word=True)
        raw_lm_score += score
        if normalized:
            raw_lm_score = raw_lm_score / (n_word + 1)
        return raw_lm_score
    
    @torch.no_grad()
    def calc_suta_loss(self, wavs):
        record = {}
        self.suta_adapt_loss_only(wavs, record=record)
        self.adapt_count -= 1
        return record

    @torch.no_grad()
    def calc_ctc_loss(self, wavs, texts):
        record = {}
        self.ctc_adapt_loss_only(wavs, texts, record=record)
        self.adapt_count -= 1
        return record

    def snapshot(self, key: str):
        """Copy the model and optimizer states for resetting after adaptation."""
        # print(f"Store state. (key: {key})")
        model_state = deepcopy(self.model.state_dict())
        optimizer_state = deepcopy(self.optimizer.state_dict())
        if self.scheduler is not None:
            scheduler_state = deepcopy(self.scheduler.state_dict())
        else:
            scheduler_state = None
        self.history[key] = (model_state, optimizer_state, scheduler_state)
    
    def load_snapshot(self, key: str) -> None:
        """Restore the model and optimizer states from copies."""
        # print(f"Reset. (key: {key})")
        model_state, optimizer_state, scheduler_state = self.history[key]
        model_state = deepcopy(model_state)
        self.model.load_state_dict(model_state, strict=True)
        
        if optimizer_state is not None:
            # optimizer_state = self.history["init"][1]
            optimizer_state = deepcopy(optimizer_state)
            self.optimizer.load_state_dict(optimizer_state)
        if scheduler_state is not None:
            scheduler_state = deepcopy(scheduler_state)
            self.scheduler.load_state_dict(scheduler_state)

    def delete_snapshot(self, key: str) -> None:
        """Delete specific history."""
        self.history.pop(key)

    def save(self, path: str) -> None:
        torch.save(self.model.state_dict(), path)

    def load(self, path: str) -> None:
        self.model.load_state_dict(torch.load(path))
    
    def collect_params(self):
        """Collect the affine scale + shift parameters from batch norms.

        Walk the model's modules and collect all batch normalization parameters.
        Return the parameters and their names.

        Note: other choices of parameterization are possible!
        """
        params = []
        names = []
        trainable = []
        if self.config["bias_only"]:
            trainable = ['bias']
        else: 
            trainable = ['weight', 'bias']

        if self.config.get("bitfit", False):
            for np, p in self.model.named_parameters():
                if str(np).split('.')[1] == 'encoder' and "bias" in np:
                    p.requires_grad = True
                    params.append(p)
                    names.append(np)
        
        for nm, m in self.model.named_modules():
            # print(nm)
            if self.config["train_LN"]: 
                if isinstance(m, nn.LayerNorm):
                    for np, p in m.named_parameters():
                        if np in trainable:
                            if not p.requires_grad:
                                p.requires_grad = True
                                params.append(p)
                                names.append(f"{nm}.{np}")
            if self.config["train_feature"]:
                if len(str(nm).split('.')) > 1:
                    if str(nm).split('.')[1] == 'feature_extractor' or str(nm).split('.')[1] == 'feature_projection':
                        for np, p in m.named_parameters():
                            p.requires_grad = True
                            params.append(p)
                            names.append(f"{nm}.{np}")
                            
            if self.config["train_all"]: 
                for np, p in m.named_parameters():
                    p.requires_grad = True
                    params.append(p)
                    names.append(f"{nm}.{np}")

        # 按参数去重（train_LN 与 train_feature 分支会把同一参数收集多份）
        seen, unique_params, unique_names = set(), [], []
        for p, n in zip(params, names):
            if id(p) not in seen:
                seen.add(id(p))
                unique_params.append(p)
                unique_names.append(n)
        return unique_params, unique_names


def setup_optimizer(params, opt_name='AdamW', lr=1e-4, beta=0.9, weight_decay=0., scheduler=None, step_size=1, gamma=0.7):
    opt = getattr(torch.optim, opt_name)
    print(f'[INFO]    optimizer: {opt}')
    print(f'[INFO]    scheduler: {scheduler}')
    if opt_name == 'Adam':       
        optimizer = opt(params,
                lr=lr,
                betas=(beta, 0.999),
                weight_decay=weight_decay)
    else: 
        optimizer = opt(params, lr=lr, weight_decay=weight_decay)
    
    if scheduler is not None: 
        return optimizer, eval(scheduler)(optimizer, step_size=step_size, gamma=gamma)
    else: 
        return optimizer, None
