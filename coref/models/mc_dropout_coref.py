import torch

from typing import List, Dict

from coref.models.general_coref_model import GeneralCorefModel
from config import Config

from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.pairwise_encoder import PairwiseEncoder
from coref.word_encoder import WordEncoder
from coref.rough_scorer import MCDropoutRoughScorer
from coref.span_predictor import SpanPredictor
from coref.bert import load_bert


class MCDropoutCorefModel(GeneralCorefModel):
    def __init__(self, config: Config, epochs_trained: int = 0):
        self.keep_dropout: List[str] = ["we", "rough_scorer"]
        super().__init__(config, epochs_trained)

    def _set_training(self, value: bool) -> None:
        self._training = value
        for name, module in self.trainable.items():
            if name not in self.keep_dropout:
                module.train(self._training)
            else:
                module.train(True)

    def _build_model(self) -> None:
        encoder, tokenizer = load_bert(
            self.config.model_params.bert_model,
            self.config.tokenizer_kwargs,
            self.config.training_params.device,
        )
        self.bert = encoder
        self.tokenizer = tokenizer
        self.pw = PairwiseEncoder(self.config).to(
            self.config.training_params.device
        )

        bert_emb = self.bert.config.hidden_size
        pair_emb = bert_emb * 3 + self.pw.shape

        # pylint: disable=line-too-long
        self.a_scorer = AnaphoricityScorer(pair_emb, self.config).to(
            self.config.training_params.device
        )
        self.we = WordEncoder(bert_emb, self.config).to(
            self.config.training_params.device
        )
        self.rough_scorer = MCDropoutRoughScorer(
            bert_emb,
            self.config.model_params.rough_k,
            self.config.training_params.dropout_rate,
            self.config.active_learning.parameters_samples,
        ).to(self.config.training_params.device)
        self.sp = SpanPredictor(bert_emb, self.config).to(
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
