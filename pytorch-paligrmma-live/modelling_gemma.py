import torch
from torch import nn 
from typing import Optional,Tuple,List
from torch.nn import CrossEntropyLoss
import math
from modeling_siglip import SiglipVisionConfig, SiglipVisionModel

class PaliGemmaForConditionalGenearation(nn.Module):
    def __init__(self,config: PaliGemmaConfig):
        super().__init__()
        self.config=config
        self.visio_tower=SiglipVisionModel(config.vision_config)
        self.multi_modal_projector=PaliGemmaMultiModalProjector(config)
        self.vocab_size=config.vocab_size

        language_model=GemmaForCasualLM(config.text_config)
        self.language_model=language_model

        self.pad_token_id=self.config.pad_token_id if self.config.pad_token_id is not None else -1
    
    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(self,
    inputs_ids:torch.LongTensor=None,
    pixel_values: torch.FloatTensor=None,
    attention_mask: Optional[torch.Tensor]=None,
    kv_cache: Optional[kVCache]=None,
    )->Tuple:
    
E=