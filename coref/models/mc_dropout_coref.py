import torch

from typing import List, Dict

from coref.models.general_coref_model import GeneralCorefModel
from config import Config
from coref.const import Doc, CorefResult

from coref.anaphoricity_scorer import MCDropoutAnaphoricityScorer
from coref.pairwise_encoder import MCDropoutPairwiseEncoder
from coref.word_encoder import WordEncoder
from coref.rough_scorer import MCDropoutRoughScorer
from coref.span_predictor import MCDropoutSpanPredictor


class MCDropoutCorefModel(GeneralCorefModel):
    def __init__(self, config: Config, epochs_trained: int = 0):
        # The approach works for all layers except of span_predictor
        # The dropout is turned on inside MCDropoutSpanPredictor.
        # TODO refactor keep _set_training from MCDropoutSpanPredictor
        self.keep_dropout: List["str"] = ["rough_scorer", "pw", "a_scorer"]
        super().__init__(config, epochs_trained)

    def _set_training(self, value: bool) -> None:
        self._training = value
        for name, module in self.trainable.items():
            if name not in self.keep_dropout:
                module.train(self._training)
            else:
                module.train(True)

    def _build_model(self) -> None:
        self.bert = self.config.model_bank.encoder
        self.tokenizer = self.config.model_bank.tokenizer
        self.pw = MCDropoutPairwiseEncoder(self.config).to(
            self.config.training_params.device
        )

        bert_emb = self.bert.config.hidden_size
        pair_emb = bert_emb * 3 + self.pw.shape

        # pylint: disable=line-too-long
        self.a_scorer = MCDropoutAnaphoricityScorer(pair_emb, self.config).to(
            self.config.training_params.device
        )
        self.we = WordEncoder(bert_emb, self.config).to(
            self.config.training_params.device
        )
        self.rough_scorer = MCDropoutRoughScorer(bert_emb, self.config).to(
            self.config.training_params.device
        )
        self.sp = MCDropoutSpanPredictor(bert_emb, self.config).to(
            self.config.training_params.device
        )

        self.trainable: Dict[str, torch.nn.Module] = {
            "bert": self.bert,
            "we": self.we,
            "rough_scorer": self.rough_scorer,
            "pw": self.pw,
            "a_scorer": self.a_scorer,
            "sp": self.sp,
        }
