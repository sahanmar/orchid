import random
from typing import Dict, List, Optional

import torch
from tqdm.auto import tqdm

from config import Config
from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.const import ReducedDimensionalityCorefResult, Doc
from coref.models.general_coref_model import GeneralCorefModel
from coref.pairwise_encoder import PairwiseEncoder
from coref.rough_scorer import RoughScorer
from coref.span_predictor import SpanPredictor
from coref.word_encoder import ReducedDimensionalityWordEncoder


class ReducedDimensionalityCorefModel(GeneralCorefModel):
    def __init__(self, config: Config, epochs_trained: int = 0):
        super(ReducedDimensionalityCorefModel, self).__init__(
            config=config,
            epochs_trained=epochs_trained,
        )

    def run(
        self,
        doc: Doc,
        normalize_anaphoras: bool = False,
    ) -> ReducedDimensionalityCorefResult:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            doc (Doc): a dictionary with the document data.
            normalize_anaphoras (bool) apply softmax or not
            to anaphoras scorer

        Returns:
            CorefResult (see const.py)
        """
        res = ReducedDimensionalityCorefResult()
        # Encode words with bert
        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        # embeddings      [n_subwords, target_emb]
        words, cluster_ids, res.manifold_learning_loss = self.we(
            doc, self._bertify(doc)
        )

        # Obtain bilinear scores and leave only top-k antecedents for each word
        # top_rough_scores  [n_words, n_ants]
        # top_indices       [n_words, n_ants]
        top_rough_scores, top_indices = self.rough_scorer(words)

        # Get pairwise features [n_words, n_ants, n_pw_features]
        pw = self.pw(top_indices, doc)

        batch_size = self.config.model_params.a_scoring_batch_size
        a_scores_lst: List[torch.Tensor] = []

        for i in range(0, len(words), batch_size):
            pw_batch = pw[i : i + batch_size]
            words_batch = words[i : i + batch_size]
            top_indices_batch = top_indices[i : i + batch_size]
            top_rough_scores_batch = top_rough_scores[i : i + batch_size]

            # a_scores_batch  [batch_size, n_ants]
            a_scores_batch = self.a_scorer(
                all_mentions=words,
                mentions_batch=words_batch,
                pw_batch=pw_batch,
                top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch,
            )
            a_scores_lst.append(a_scores_batch)

        # coref_scores   [n_spans, n_ants]
        cat_anaphora_scores = torch.cat(a_scores_lst, dim=0)
        res.coref_scores = (
            torch.softmax(cat_anaphora_scores, dim=1)
            if normalize_anaphoras
            else cat_anaphora_scores
        )

        res.coref_y = self._get_ground_truth(
            cluster_ids, top_indices, (top_rough_scores > float("-inf"))
        )
        res.word_clusters = self._clusterize(doc, res.coref_scores, top_indices)
        res.span_scores, res.span_y = self.sp.get_training_data(doc, words)

        if not self.training:
            res.span_clusters = self.sp.predict(doc, words, res.word_clusters)

        return res

    def train(
        self, docs: List[Doc], docs_dev: Optional[List[Doc]] = None
    ) -> None:
        """
        Trains all the trainable blocks in the model using the config provided.
        """
        docs_ids = list(range(len(docs)))
        avg_spans = sum(len(doc.head2span) for doc in docs) / len(docs)

        for epoch in range(
            self.epochs_trained, self.config.training_params.train_epochs
        ):
            self.training = True
            running_c_loss = 0.0
            running_s_loss = 0.0
            running_emb_loss = 0.0
            random.shuffle(docs_ids)
            pbar = tqdm(docs_ids, unit="docs", ncols=0)
            for doc_id in pbar:
                doc = docs[doc_id]

                for optim in self.optimizers.values():
                    optim.zero_grad()

                res = self.run(doc)

                # Evaluate losses
                c_loss = self._coref_criterion(res.coref_scores, res.coref_y)
                emb_loss = res.manifold_learning_loss
                assert emb_loss is not None, (
                    f"Manifold Learning loss is not allowed to be "
                    f"None during training"
                )
                if res.span_y and res.span_scores is not None:
                    s_loss = (
                        (
                            self._span_criterion(
                                res.span_scores[:, :, 0], res.span_y[0]
                            )
                            + self._span_criterion(
                                res.span_scores[:, :, 1], res.span_y[1]
                            )
                        )
                        / avg_spans
                        / 2
                    )
                else:
                    s_loss = torch.zeros_like(c_loss)

                del res

                (c_loss + s_loss + emb_loss).backward()
                running_c_loss += c_loss.item()
                running_s_loss += s_loss.item()
                running_emb_loss += emb_loss

                del c_loss, s_loss, emb_loss

                for optim in self.optimizers.values():
                    optim.step()
                for scheduler in self.schedulers.values():
                    scheduler.step()

                pbar.set_description(
                    f"Epoch {epoch + 1}:"
                    f" {doc.document_id:26}"
                    f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                    f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
                    f" emb_loss: {running_emb_loss / (pbar.n + 1):<.5f}"
                )

            self.epochs_trained += 1
            self.save_weights()
            if docs_dev is not None:
                self.evaluate(docs=docs_dev)

    def _build_model(self) -> None:
        self.bert = self.config.model_bank.encoder
        self.tokenizer = self.config.model_bank.tokenizer
        self.pw = PairwiseEncoder(self.config).to(
            self.config.training_params.device
        )

        bert_emb = self.bert.config.hidden_size
        self.we = ReducedDimensionalityWordEncoder(
            features=bert_emb,
            config=self.config,
        ).to(self.config.training_params.device)

        self.rough_scorer = RoughScorer(self.we.features_out, self.config).to(
            self.config.training_params.device
        )

        pair_emb = self.we.features_out * 3 + self.pw.shape
        self.a_scorer = AnaphoricityScorer(pair_emb, self.config).to(
            self.config.training_params.device
        )

        self.sp = SpanPredictor(self.we.features_out, self.config).to(
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
