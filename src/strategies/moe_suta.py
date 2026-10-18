from torch.utils.data import Dataset
import yaml
from tqdm import tqdm

from ..system.suta import SUTASystem
from ..utils.tool import wer
from .base import IStrategy


class MOE_SUTAStrategy(IStrategy):
    def __init__(self, config) -> None:
        self.config = config
        self.strategy_config = config["strategy_config"]
        self.system = SUTASystem(config["system_config"])
        self.system.eval()
    
    def _init_start(self, sample) -> None:
        # EXP-G：连续适应（CSUTA 式），叠加在 EXP-F 的 SUTA 原版 loss 之上
        pass
    
    def _adapt(self, sample):
        self.system.eval()
        is_collapse = False
        for _ in range(self.strategy_config["steps"]):
            record = {}
            self.system.moe_suta_adapt(
                wavs=[sample["wav"]],
                record=record,
            )
            if record.get("collapse", False):
                is_collapse = True
        if is_collapse:
            print("oh no")

    def _update(self, sample):
        pass
    
    def run(self, ds: Dataset):
        long_cnt = 0
        n_words = []
        errs, losses = [], []
        transcriptions = []
        for sample in tqdm(ds):
            if len(sample["wav"]) > self.strategy_config["max_length"]:
                long_cnt += 1
                continue
            n_words.append(len(sample["text"].split(" ")))

            self._init_start(sample)
            self._adapt(sample)

            self.system.eval()
            trans = self.system.inference([sample["wav"]])[0]
            err = wer(sample["text"], trans)
            errs.append(err)
            transcriptions.append((sample["text"], trans))
            
            # loss
            # loss = self.system.calc_suta_loss([sample["wav"]])
            # ctc_loss = self.system.calc_ctc_loss([sample["wav"]], [sample["text"]])
            # loss["ctc_loss"] = ctc_loss["ctc_loss"]
            # losses.append(loss)

            self._update(sample)
            
        print("#Too long: ", long_cnt)
        
        return {
            "wers": errs,
            "n_words": n_words,
            "transcriptions": transcriptions,
            "losses": losses,
        }
    
    def get_adapt_count(self):
        return self.system.adapt_count

    def load_checkpoint(self, path):
        self.system.load(path)