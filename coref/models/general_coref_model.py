import os
import random
import re
from datetime import datetime
from typing import (
    Any,
    Dict,
    List,
    Optional,
    Set,
    Tuple,
    Hashable,
    cast,
    TYPE_CHECKING,
    TypeVar,
    Union,
    Callable,
)
from copy import deepcopy

import numpy as np  # type: ignore
import torch
import transformers  # type: ignore
from tqdm import tqdm

from coref import bert, conll, utils
from coref.anaphoricity_scorer import AnaphoricityScorer
from coref.cluster_checker import ClusterChecker
from config import Config
from config.active_learning import InstanceSampling, SamplingStrategy
from coref.const import CorefResult, Doc, SampledData
from coref.loss import CorefLoss
from coref.pairwise_encoder import PairwiseEncoder
from coref.rough_scorer import RoughScorer
from coref.span_predictor import SpanPredictor
from coref.utils import GraphNode
from coref.word_encoder import WordEncoder
from uncertainty.uncertainty_metrics import pavpu_metric
from coref.logging_utils import get_stream_logger

if TYPE_CHECKING:
    from active_learning.exploration import GreedySampling, NaiveSampling

T = TypeVar("T")


class GeneralCorefModel:  # pylint: disable=too-many-instance-attributes
    """Combines all coref modules together to find coreferent spans.

    Attributes:
        config (config.Config): the model's configuration,
            see config.toml for the details
        epochs_trained (int): number of epochs the model has been trained for

    Submodules (in the order of their usage in the pipeline):
        tokenizer (transformers.AutoTokenizer)
        bert (transformers.AutoModel)
        we (WordEncoder)
        rough_scorer (RoughScorer)
        pw (PairwiseEncoder)
        a_scorer (AnaphoricityScorer)
        sp (SpanPredictor)
    """

    def __init__(self, config: Config, epochs_trained: int = 0):
        """
        A newly created model is set to evaluation mode.

        Args:
            config (Config): config dataclass created from toml
            epochs_trained (int): the number of epochs finished
                (useful for warm start)
        """
        self.config = config

        self._logger = get_stream_logger(
            logging_conf=self.config.logging,
            experiment=self.config.model_params.bert_model,
        )
        self._logger.info(f"Initializing the general coreference model")

        self.epochs_trained = epochs_trained
        self._docs: Dict[str, List[Doc]] = {}
        self._build_model()
        self._build_optimizers()
        self._set_training(False)
        self._build_criteria()

        # Active Learning section
        self.sampling_strategy_config: Union[
            GreedySampling, NaiveSampling
        ] = config.active_learning.sampling_strategy
        self.sampling_strategy_type: SamplingStrategy = (
            config.active_learning.strategy_type
        )
        self._logger.info(
            "Initialization of the general coreference model is complete"
        )

    @property
    def training(self) -> bool:
        """Represents whether the model is in the training mode"""
        return self._training

    @training.setter
    def training(self, new_value: bool) -> None:
        if self._training is new_value:
            return
        self._set_training(new_value)

    # ========================================================== Public methods

    @torch.no_grad()
    def evaluate(
        self, docs: List[Doc], word_level_conll: bool = False
    ) -> Tuple[float, float, float, float]:
        """Evaluates the modes on the data split provided.

        Args:
            data_split (str): one of 'dev'/'test'/'train'
            word_level_conll (bool): if True, outputs conll files on word-level

        Returns:
            mean loss
            span-level LEA: f1, precision, recal
        """
        self.training = False
        w_checker = ClusterChecker()
        s_checker = ClusterChecker()
        running_loss = 0.0
        s_correct = 0
        s_total = 0

        # TODO think about the hardcoded 'dev'
        with conll.open_(self.config, self.epochs_trained, "dev") as (
            gold_f,
            pred_f,
        ):
            pbar = tqdm(docs, unit="docs", ncols=0)
            for doc in pbar:
                res = cast(CorefResult, self.run(deepcopy(doc), True))

                running_loss += self._coref_criterion(
                    res.coref_scores, res.coref_y
                ).item()

                if (
                    res.span_y is not None
                    and res.span_scores is not None
                    and res.span_y
                ):
                    pred_starts = res.span_scores[:, :, 0].argmax(dim=1)
                    pred_ends = res.span_scores[:, :, 1].argmax(dim=1)
                    s_correct += (
                        (
                            (res.span_y[0] == pred_starts)
                            * (res.span_y[1] == pred_ends)
                        )
                        .sum()
                        .item()
                    )
                    s_total += len(pred_starts)

                if word_level_conll:
                    if res.word_clusters is None:
                        raise RuntimeError(
                            f'"word_clusters" attribute must be set'
                        )
                    conll.write_conll(
                        doc,
                        [
                            [(i, i + 1) for i in cluster]
                            for cluster in doc.word_clusters
                        ],
                        gold_f,
                    )
                    conll.write_conll(
                        doc,
                        [
                            [(i, i + 1) for i in cluster]
                            for cluster in res.word_clusters
                        ],
                        pred_f,
                    )
                else:
                    conll.write_conll(doc, doc.span_clusters, gold_f)
                    if res.span_clusters is None:
                        raise RuntimeError(
                            f'"span_clusters" attribute must be set'
                        )
                    conll.write_conll(doc, res.span_clusters, pred_f)

                w_checker.add_predictions(
                    doc.word_clusters,
                    cast(List[List[Hashable]], res.word_clusters),
                )
                w_lea = w_checker.total_lea

                s_checker.add_predictions(
                    doc.span_clusters,
                    cast(List[List[Hashable]], res.span_clusters),
                )
                s_lea = s_checker.total_lea

                del res

                pbar.set_description(
                    f"{'test'}:"
                    f" | WL: "
                    f" loss: {running_loss / (pbar.n + 1):<.5f},"
                    f" f1: {w_lea[0]:.5f},"
                    f" p: {w_lea[1]:.5f},"
                    f" r: {w_lea[2]:<.5f}"
                    f" | SL: "
                    f" sa: {s_correct / s_total:<.5f},"
                    f" f1: {s_lea[0]:.5f},"
                    f" p: {s_lea[1]:.5f},"
                    f" r: {s_lea[2]:<.5f}"
                )
            print()

        avg_loss = float(running_loss / len(docs))
        tot_f1, tot_prec, tot_rec = s_checker.total_lea

        self._logger.info(
            f"EVAL METRICS | loss: {avg_loss:<.5f} | f1: {tot_f1:.5f} prec: {tot_prec:.5f} recall: {tot_rec:.5f}\n"
        )

        return (
            float(running_loss / len(docs)),
            *s_checker.total_lea,
        )

    def load_weights(
        self,
        path: Optional[str] = None,
        ignore: Optional[Set[str]] = None,
        map_location: Optional[str] = None,
        noexception: bool = False,
    ) -> None:
        """
        Loads pretrained weights of modules saved in a file located at path.
        If path is None, the last saved model with current configuration
        in data_dir is loaded.
        Assumes files are named like {configuration}_(e{epoch}_{time})*.pt.
        """
        if path is None:
            pattern = rf"{self.config.section}_\(e(\d+)_[^()]*\).*\.pt"
            files = []
            for f in os.listdir(self.config.data.data_dir):
                match_obj = re.match(pattern, f)
                if match_obj:
                    files.append((int(match_obj.group(1)), f))
            if not files:
                if noexception:
                    self._logger.info("No weights have been loaded")
                    return
                raise OSError(
                    f"No weights found in {self.config.data.data_dir}!"
                )
            _, path = sorted(files)[-1]
            path = os.path.join(self.config.data.data_dir, path)

        if map_location is None:
            map_location = self.config.training_params.device
        self._logger.info(f"Loading weights from {path}...")
        state_dicts = torch.load(path, map_location=map_location)
        self.epochs_trained = state_dicts.pop("epochs_trained", 0)
        for key, state_dict in state_dicts.items():
            if not ignore or key not in ignore:
                if key.endswith("_optimizer"):
                    self.optimizers[key].load_state_dict(state_dict)
                elif key.endswith("_scheduler"):
                    self.schedulers[key].load_state_dict(state_dict)
                else:
                    self.trainable[key].load_state_dict(state_dict)
                self._logger.info(f"Loaded {key}")

    def run(
        self,
        doc: Doc,
        normalize_anaphoras: bool = False,
        return_mention: bool = False,
        scoring_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ) -> Union[CorefResult, list[Tuple[int, float]]]:
        """
        This is a massive method, but it made sense to me to not split it into
        several ones to let one see the data flow.

        Args:
            doc (Doc): a dataframe with the document data.
            normalize_anaphoras (bool): apply softmax or not
            to anaphoras scorer
            return_mentions (bool): return only rough scores

        Returns:
            CorefResult (see const.py)
        """

        # Encode words with bert
        encoded_doc = self._bertify(doc)

        # If instance_sampling is `token` or `mention`, then rewrite the doc to a pseudo doc. This will use
        # only tokens, given in the simulation_token_annotations field. If the field
        # is empty, the method will use all available annotated tokens.
        # N.B. The quality of encoding is not damaged because it is done on the whole
        # article
        if (
            self.config.active_learning.instance_sampling
            in {
                InstanceSampling.random_token,
                InstanceSampling.random_mention,
            }
            and not return_mention
        ):
            doc = doc.create_simulation_pseudodoc()
            encoded_doc = encoded_doc[
                doc.simulation_token_annotations.original_subtokens_ids, :
            ]

        # words           [n_words, span_emb]
        # cluster_ids     [n_words]
        words, cluster_ids = self.we(doc, encoded_doc)

        # If scoring_fn is None
        #     Obtain bilinear scores and leave only top-k antecedents for each word
        #     rough_scores         [n_words, n_ants]
        #     rough_scores_indices [n_words, n_ants]
        # Else
        #     sorted scores given scoring fn and their original  positions
        #     rough_scores         [n_words, 1]
        #     rough_scores_indices [n_words, 1]
        rough_scores, rough_scores_indices = cast(
            Tuple[torch.Tensor, torch.Tensor],
            self.rough_scorer(words, scoring_fn),
        )
        if return_mention:
            return doc.subwords_2_words_w_payload(
                [int(i) for i in rough_scores_indices.flatten().tolist()],
                rough_scores.flatten().tolist(),
            )

        # Get pairwise features [n_words, n_ants, n_pw_features]
        pw = self.pw(rough_scores_indices, doc)

        batch_size = self.config.model_params.a_scoring_batch_size
        a_scores_lst: List[torch.Tensor] = []

        for i in range(0, len(words), batch_size):
            pw_batch = pw[i : i + batch_size]
            words_batch = words[i : i + batch_size]
            top_indices_batch = rough_scores_indices[i : i + batch_size]
            top_rough_scores_batch = rough_scores[i : i + batch_size]

            # a_scores_batch  [batch_size, n_ants]
            a_scores_batch = self.a_scorer(
                all_mentions=words,
                mentions_batch=words_batch,
                pw_batch=pw_batch,
                top_indices_batch=top_indices_batch,
                top_rough_scores_batch=top_rough_scores_batch,
            )
            a_scores_lst.append(a_scores_batch)

        res = CorefResult()

        # coref_scores   [n_spans, n_ants]
        cat_anaphora_scores = torch.cat(a_scores_lst, dim=0)
        res.coref_scores = (
            torch.softmax(cat_anaphora_scores, dim=1)
            if normalize_anaphoras
            else cat_anaphora_scores
        )

        res.coref_y = self._get_ground_truth(
            cluster_ids, rough_scores_indices, (rough_scores > float("-inf"))
        )
        res.word_clusters = self._clusterize(
            doc, res.coref_scores, rough_scores_indices
        )
        res.span_scores, res.span_y = self.sp.get_training_data(doc, words)

        if not self.training:
            res.span_clusters = self.sp.predict(doc, words, res.word_clusters)

        return res

    def save_weights(self) -> None:
        """Saves trainable models as state dicts."""
        to_save: List[Tuple[str, Any]] = [
            (key, value)
            for key, value in self.trainable.items()
            if self.config.training_params.bert_finetune or key != "bert"
        ]
        to_save.extend(self.optimizers.items())
        to_save.extend(self.schedulers.items())

        time = datetime.strftime(datetime.now(), "%Y.%m.%d_%H.%M")
        path = os.path.join(
            self.config.data.data_dir,
            f"{self.config.section}" f"_(e{self.epochs_trained}_{time}).pt",
        )
        savedict = {name: module.state_dict() for name, module in to_save}
        savedict["epochs_trained"] = self.epochs_trained  # type: ignore
        torch.save(savedict, path)

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
            random.shuffle(docs_ids)
            pbar = tqdm(docs_ids, unit="docs", ncols=0)
            for doc_id in pbar:
                doc = docs[doc_id]

                for optim in self.optimizers.values():
                    optim.zero_grad()

                res = cast(CorefResult, self.run(deepcopy(doc)))

                c_loss = self._coref_criterion(res.coref_scores, res.coref_y)
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

                (c_loss + s_loss).backward()
                running_c_loss += c_loss.item()
                running_s_loss += s_loss.item()

                del c_loss, s_loss

                for optim in self.optimizers.values():
                    optim.step()
                for scheduler in self.schedulers.values():
                    scheduler.step()

                pbar.set_description(
                    f"Epoch {epoch + 1}:"
                    f" {doc.document_id:26}"
                    f" c_loss: {running_c_loss / (pbar.n + 1):<.5f}"
                    f" s_loss: {running_s_loss / (pbar.n + 1):<.5f}"
                )

            self.epochs_trained += 1
            self.save_weights()
            if docs_dev is not None:
                self._logger.info(f"TRAINING | epoch {epoch} is finished")
                self.evaluate(docs=docs_dev)

    def sample_unlabled_data(self, documents: List[Doc]) -> SampledData:
        if (
            self.config.active_learning.instance_sampling
            == InstanceSampling.random_mention
        ):
            mentions = {
                doc.orchid_id: self.run(deepcopy(doc), return_mention=True)
                for doc in documents
            }
        elif (
            self.config.active_learning.instance_sampling
            == InstanceSampling.entropy_mention
        ):
            ...
        else:
            mentions = {}

        if SamplingStrategy.greedy_sampling == self.sampling_strategy_type:
            return self.sampling_strategy_config.step(documents)  # type: ignore
        if SamplingStrategy.naive_sampling == self.sampling_strategy_type:
            return self.sampling_strategy_config.step(  # type: ignore
                documents, cast(dict[str, list[int]], mentions)
            )

        raise ValueError("Wrong sampling strategy... Executor not likey...")

    def get_uncertainty_metrics(self, docs: List[Doc]) -> list[float]:

        metrics_vals: list[list[float]] = []
        pbar = tqdm(docs, unit="docs", ncols=0)
        for doc in pbar:
            res = cast(CorefResult, self.run(deepcopy(doc), True))
            pavpu_output = pavpu_metric(
                res.coref_scores, res.coref_y, self.config.metrics.pavpu
            )
            metrics_vals.append(pavpu_output)

        metrics_val: list[float] = np.mean(
            np.array(metrics_vals), axis=0
        ).tolist()

        self._logger.info(f"PAVPU METRICS | avg_pavpu: {metrics_val}")

        return metrics_val

    # ========================================================= Private methods

    def _bertify(self, doc: Doc) -> torch.Tensor:
        subwords_batches = bert.get_subwords_batches(
            doc, self.tokenizer, self.config.model_params.bert_window_size
        )

        special_tokens = np.array(
            [
                self.tokenizer.cls_token_id,
                self.tokenizer.sep_token_id,
                self.tokenizer.pad_token_id,
            ]
        )
        subword_mask = ~(np.isin(subwords_batches, special_tokens))

        subwords_batches_tensor = torch.tensor(
            subwords_batches,
            device=self.config.training_params.device,
            dtype=torch.long,
        )
        subword_mask_tensor = torch.tensor(
            subword_mask, device=self.config.training_params.device
        )

        # Obtain bert output for selected batches only
        attention_mask = subwords_batches != self.tokenizer.pad_token_id
        out = self.bert(
            subwords_batches_tensor,
            attention_mask=torch.tensor(
                attention_mask, device=self.config.training_params.device
            ),
        )

        # [n_subwords, bert_emb]
        return out.last_hidden_state[subword_mask_tensor]

    def _build_model(self) -> None:
        self.bert = self.config.model_bank.encoder
        self.tokenizer = self.config.model_bank.tokenizer
        self.pw = PairwiseEncoder(self.config).to(
            self.config.training_params.device
        )

        bert_emb = self.bert.config.hidden_size
        pair_emb = bert_emb * 3 + self.pw.shape

        # pylint: disable=line-too-long
        self.a_scorer = AnaphoricityScorer(pair_emb, self.config).to(
            self.config.training_params.device
        )
        self.we = WordEncoder(self.config).to(
            self.config.training_params.device
        )
        self.rough_scorer = RoughScorer(
            bert_emb,
            self.config.model_params.rough_k,
            self.config.training_params.dropout_rate,
        ).to(self.config.training_params.device)
        self.sp = SpanPredictor(self.config).to(
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

    def _build_criteria(self) -> None:
        self._coref_criterion = CorefLoss(
            self.config.training_params.bce_loss_weight
        )
        self._span_criterion = torch.nn.CrossEntropyLoss(reduction="sum")

    def _build_optimizers(self) -> None:
        # This is very bad. Caching the entire dataset in order to get
        # the number of docs.
        # TODO see if this doesn't break smth
        # n_docs = len(self._get_docs(self.config.train_data))
        n_docs = self.config.data.num_of_training_docs
        self.optimizers: Dict[str, torch.optim.Optimizer] = {}
        self.schedulers: Dict[str, torch.optim.lr_scheduler.LambdaLR] = {}

        for param in self.bert.parameters():
            param.requires_grad = self.config.training_params.bert_finetune

        if self.config.training_params.bert_finetune:
            self.optimizers["bert_optimizer"] = torch.optim.Adam(
                self.bert.parameters(),
                lr=self.config.training_params.bert_learning_rate,
            )
            self.schedulers[
                "bert_scheduler"
            ] = transformers.get_linear_schedule_with_warmup(
                self.optimizers["bert_optimizer"],
                n_docs,
                n_docs * self.config.training_params.train_epochs,
            )

        # Must ensure the same ordering of parameters between launches
        modules = sorted(
            (key, value)
            for key, value in self.trainable.items()
            if key != "bert"
        )
        params = []
        for _, module in modules:
            for param in module.parameters():
                param.requires_grad = True
                params.append(param)

        self.optimizers["general_optimizer"] = torch.optim.Adam(
            params, lr=self.config.training_params.learning_rate
        )
        self.schedulers[
            "general_scheduler"
        ] = transformers.get_linear_schedule_with_warmup(
            self.optimizers["general_optimizer"],
            0,
            n_docs * self.config.training_params.train_epochs,
        )

    def _clusterize(
        self, doc: Doc, scores: torch.Tensor, top_indices: torch.Tensor
    ) -> List[List[int]]:
        antecedents = scores.argmax(dim=1) - 1
        not_dummy = antecedents >= 0
        coref_span_heads = torch.arange(0, len(scores))[not_dummy]
        antecedents = top_indices[coref_span_heads, antecedents[not_dummy]]

        nodes = [GraphNode(i) for i in range(len(doc.cased_words))]
        for i, j in zip(coref_span_heads.tolist(), antecedents.tolist()):
            nodes[i].link(nodes[j])
            assert nodes[i] is not nodes[j]

        clusters: List[List[int]] = []
        for node in nodes:
            if len(node.links) > 0 and not node.visited:
                cluster: List[int] = []
                stack = [node]
                while stack:
                    current_node = stack.pop()
                    current_node.visited = True
                    cluster.append(current_node.id)
                    stack.extend(
                        link for link in current_node.links if not link.visited
                    )
                assert len(cluster) > 1
                clusters.append(sorted(cluster))
        return sorted(clusters)

    @staticmethod
    def _get_ground_truth(
        cluster_ids: torch.Tensor,
        top_indices: torch.Tensor,
        valid_pair_map: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            cluster_ids: tensor of shape [n_words], containing cluster indices
                for each word. Non-gold words have cluster id of zero.
            top_indices: tensor of shape [n_words, n_ants],
                indices of antecedents of each word
            valid_pair_map: boolean tensor of shape [n_words, n_ants],
                whether for pair at [i, j] (i-th word and j-th word)
                j < i is True

        Returns:
            tensor of shape [n_words, n_ants + 1] (dummy added),
                containing 1 at position [i, j] if i-th and j-th words corefer.
        """
        y = cluster_ids[top_indices] * valid_pair_map  # [n_words, n_ants]
        y[y == 0] = -1  # -1 for non-gold words
        y = utils.add_dummy(y)  # [n_words, n_cands + 1]
        y = y == cluster_ids.unsqueeze(1)  # True if coreferent
        # For all rows with no gold antecedents setting dummy to True
        y[y.sum(dim=1) == 0, 0] = True
        return y.to(torch.float)

    def _set_training(self, value: bool) -> None:
        self._training = value
        for module in self.trainable.values():
            module.train(self._training)
