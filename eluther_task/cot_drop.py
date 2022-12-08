"""
DROP: A Reading Comprehension Benchmark Requiring Discrete Reasoning Over Paragraphs
https://aclanthology.org/attachments/N19-1246.Supplementary.pdf

DROP is a QA dataset which tests comprehensive understanding of paragraphs. In 
this crowdsourced, adversarially-created, 96k question-answering benchmark, a 
system must resolve multiple references in a question, map them onto a paragraph,
and perform discrete operations over them (such as addition, counting, or sorting).

Homepage: https://allenai.org/data/drop

Acknowledgement: This implementation is based on the official evaluation for `DROP`:
https://github.com/allenai/allennlp-reading-comprehension/blob/master/allennlp_rc/eval/drop_eval.py
"""
import inspect
import numpy as np
import re
import json
import string
import lm_eval.datasets.drop.drop
from scipy.optimize import linear_sum_assignment
from lm_eval.base import Task, rf
from lm_eval.metrics import mean
from lm_eval import utils
from num2words import num2words
import faiss
import sys
sys.path.insert(0, '<path_to_set_repo>')
import sentence_transformers

_ARTICLES = re.compile(r"\b(a|an|the)\b", re.UNICODE)


class DROP(Task):
    VERSION = 1
    DATASET_PATH = inspect.getfile(lm_eval.datasets.drop.drop)
    DATASET_NAME = None

    def __init__(self, data_dir=None, cache_dir=None, download_mode=None, **kwargs):
        self.train_file = kwargs["train_file"]
        self.validation_file = kwargs["validation_file"]
        self.dataset = self.load_dataset()
        self._training_docs = None
        self._fewshot_docs = None
        self.nn_model = sentence_transformers.SentenceTransformer('quora-distilbert-multilingual')
        quantizer = faiss.IndexFlatIP(768)
        self.index = faiss.IndexIVFFlat(quantizer, 768, 6, faiss.METRIC_INNER_PRODUCT)
        self.load_clustering_model()

    def load_clustering_model(self):
        corpus_sentences = []
        for blob in self.dataset["train"]:
            corpus_sentences.append(blob["question"])
        corpus_embeddings = self.nn_model.encode(corpus_sentences, show_progress_bar=True, convert_to_numpy=True)
        corpus_embeddings = corpus_embeddings / np.linalg.norm(corpus_embeddings, axis=1)[:, None]
        self.index.train(corpus_embeddings)
        self.index.add(corpus_embeddings)

    def load_dataset(self):
        dataset_dict = {"train": [], "validation": []}
        dataset_dict["train"] = self.read_dataset(self.train_file))
        dataset_dict["validation"] = self.read_dataset(self.validation_file)
        return dataset_dict

    def read_dataset(self, filepath):
        blobs = []
        data = json.load(open(filepath))
        for _, psg_info in data.items():
            for qap in psg_info["qa_pairs"]:
                blobs.append({"id": qap["query_id"], "passage": psg_info["passage"], "question": qap["question"],
                                    "answers": self.parse_answer(qap["answer"])})
        return blobs

    def training_docs(self):
        docs = {}
        for qap in self.dataset["train"]:
            docs[qap["id"]] = qap
        return docs

    def validation_docs(self):
        return self.dataset["validation"]

    def has_training_docs(self):
        return True

    def has_validation_docs(self):
        return True

    def has_test_docs(self):
        return False


    def _process_doc(self, doc):
        return {
            "id": doc["query_id"],
            "passage": doc["passage"],
            "question": doc["question"],
            "answers": self.get_answers(doc),
        }

     def fewshot_examples(self, k, rnd, doc=None):
        if self._training_docs is None:
            self._training_docs = self.training_docs()

        dev_q = doc["question"]
        question_embedding = self.nn_model.encode(dev_q)
        question_embedding = question_embedding / np.linalg.norm(question_embedding)
        question_embedding = np.expand_dims(question_embedding, axis=0)
        distances, corpus_ids = self.index.search(question_embedding, 2 * k)

        sampled_docs = []
        doc_ids = rnd.sample(corpus_ids[0].tolist(), k)
        for corpus_id in doc_ids:
            t_doc = self.dataset["train"][corpus_id]
            if t_doc["id"] == doc["id"]:
                continue
            sampled_docs.append(t_doc)

        return sampled_docs


    @utils.positional_deprecated
    def fewshot_context(self, doc, num_fewshot, provide_description=None, rnd=None, description=None):
        """ Returns a fewshot context string that is made up of a prepended description
        (if provided), the `num_fewshot` number of examples, and an appended prompt example.

        :param doc: str
            The document as returned from training_docs, validation_docs, or test_docs.
        :param num_fewshot: int
            The number of fewshot examples to provide in the returned context string.
        :param provide_description: bool
            Not implemented, and this option is deprecated and will be removed in a future version in favor of a different description providing method
        :param rnd: random.Random
            The pseudo-random number generator used to randomly sample examples.
            WARNING: This is currently a required arg although it's optionalized with a default `None`.
        :param description: str
            The task's description that will be prepended to the fewshot examples.
        :returns: str
            The fewshot context.
        """
        assert rnd is not None, "A `random.Random` generator argument must be provided to `rnd`"
        assert not provide_description, (
            "The `provide_description` arg will be removed in future versions. To prepend "
            "a custom description to the context, supply the corresponding string via the "
            "`description` arg."
        )
        if provide_description is not None:
            # nudge people to not specify it at all
            print(
                "WARNING: provide_description is deprecated and will be removed in a future version in favor of description_dict")

        description = description + "\n\n" if description else ""

        if num_fewshot == 0:
            labeled_examples = ""
        else:
            # for sets with no training docs, draw from other set *but ensure no overlap with current doc*
            if self.has_training_docs():
                fewshotex = self.fewshot_examples(k=num_fewshot, rnd=rnd, doc=doc)
            else:
                if self._fewshot_docs is None:
                    self._fewshot_docs = list(
                        self.validation_docs() if self.has_validation_docs() else self.test_docs()
                    )

                fewshotex = rnd.sample(self._fewshot_docs, num_fewshot + 1)

                # get rid of the doc that's the one we're evaluating, if it's in the fewshot
                fewshotex = [x for x in fewshotex if x != doc][:num_fewshot]

            labeled_examples = "\n\n".join(
                [self.doc_to_text(doc) + self.doc_to_target(doc) for doc in fewshotex]
            ) + "\n\n"

        example = self.doc_to_text(doc)

        return description + labeled_examples + example

    @classmethod
    def get_answers(cls, qa):
        def _flatten_validated_answers(validated_answers):
            """ Flattens a dict of lists of validated answers.
            {"number": ['1', '8'], ...}
            -> [{"number": ['1'], ...}, {"number": ['8'], ...}]
            """
            vas = []
            for i in range(len(validated_answers["number"])):
                vas.append({
                    "number": validated_answers["number"][i],
                    "date": validated_answers["date"][i],
                    "spans": validated_answers["spans"][i],
                })
            return vas
        answers = []
        answers_set = set()
        candidates = [qa["answer"]] + _flatten_validated_answers(qa["validated_answers"])
        for candidate in candidates:
            answer = cls.parse_answer(candidate)
            if answer in answers_set:
                continue
            answers_set.add(answer)
            answers.append(answer)
        return answers

    @classmethod
    def parse_answer(cls, answer):
        # NOTE: Everything is returned as a tuple for uniformity and hashability.
        if answer["number"] != "":
            return (str(answer["number"]),)
        if answer["spans"] != []:
            return tuple(answer["spans"])
        return (" ".join([answer["date"]["day"],
                          answer["date"]["month"],
                          answer["date"]["year"]]).strip(),)

    def doc_to_text(self, doc):
        return f"Passage: {doc['passage']}\nQuestion: {doc['question']}\nAnswer:"

    def doc_to_text_decomp(self, doc):
        fixed_hints = f"Passage: {doc['passage']}\nComplex: {doc['question']}\n"
        input_str = fixed_hints
        for dec_q, dec_a in doc["decomposition"]:
            input_str = f"{input_str} Simple: {dec_q} Answer: {dec_a}\n"
        return input_str

    def doc_to_target(self, doc):
        return " " + ", ".join(doc["answers"])

    def construct_requests(self, doc, ctx):
        """Uses RequestFactory to construct Requests and returns an iterable of
        Requests which will be sent to the LM.

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param ctx: str
            The context string, generated by fewshot_context. This includes the natural
            language description, as well as the few shot examples, and the question
            part of the document for `doc`.
        """
        conts = [rf.greedy_until(ctx, ["."])]
        return conts

    def process_results(self, doc, results):
        """Take a single document and the LM results and evaluates, returning a
        dict where keys are the names of submetrics and values are the values of
        the metric for that one document

        :param doc:
            The document as returned from training_docs, validation_docs, or test_docs.
        :param results:
            The results of the requests created in construct_requests.
        """
        preds, golds = results, doc["answers"]
        max_em = 0
        max_f1 = 0
        preds = preds[0].strip().split('\n')[0].strip()
        for gold_answer in golds:
            if gold_answer[0].strip():
                exact_match, f1_score = self.get_metrics(preds, gold_answer)
                max_em = max(max_em, exact_match)
                max_f1 = max(max_f1, f1_score)
        print(json.dumps({"id": doc['id'], "f1": max_f1,
                          "prediction": preds, "gold": golds, "task": "drop"}))
        return {
            "em": max_em,
            "f1": max_f1
        }


    def get_prediction(self, doc, results):
        preds, golds = results, doc["answers"]
        max_em = 0
        max_f1 = 0
        preds = preds[0].strip().split('\n')[0].strip()
        for gold_answer in golds:
            if gold_answer[0].strip():
                exact_match, f1_score = self.get_metrics(preds, gold_answer)
                max_em = max(max_em, exact_match)
                max_f1 = max(max_f1, f1_score)
        return {
            "em": max_em,
            "f1": max_f1,
            "pred": preds
        }

    def get_metrics(self, predicted, gold):
        """
        Takes a predicted answer and a gold answer (that are both either a string or a list of
        strings), and returns exact match and the DROP F1 metric for the prediction.  If you are
        writing a script for evaluating objects in memory (say, the output of predictions during
        validation, or while training), this is the function you want to call, after using
        :func:`answer_json_to_strings` when reading the gold answer from the released data file.
        """
        predicted_bags = self._answer_to_bags(predicted)
        gold_bags = self._answer_to_bags(gold)

        if set(predicted_bags[0]) == set(gold_bags[0]) and len(predicted_bags[0]) == len(gold_bags[0]):
            exact_match = 1.0
        else:
            exact_match = 0.0

        f1_per_bag = self._align_bags(predicted_bags[1], gold_bags[1])
        f1 = np.mean(f1_per_bag)
        f1 = round(f1, 2)
        return exact_match, f1

    def _answer_to_bags(self, answer):
        if isinstance(answer, (list, tuple)):
            raw_spans = answer
        else:
            raw_spans = [answer]
        normalized_spans = []
        token_bags = []
        for raw_span in raw_spans:
            normalized_span = self._normalize(raw_span)
            normalized_spans.append(normalized_span)
            token_bags.append(set(normalized_span.split()))
        return normalized_spans, token_bags

    def _align_bags(self, predicted, gold):
        """
        Takes gold and predicted answer sets and first finds the optimal 1-1 alignment
        between them and gets maximum metric values over all the answers.
        """
        scores = np.zeros([len(gold), len(predicted)])
        for gold_index, gold_item in enumerate(gold):
            for pred_index, pred_item in enumerate(predicted):
                if self._match_numbers_if_present(gold_item, pred_item):
                    scores[gold_index, pred_index] = self._compute_f1(pred_item, gold_item)
        row_ind, col_ind = linear_sum_assignment(-scores)

        max_scores = np.zeros([max(len(gold), len(predicted))])
        for row, column in zip(row_ind, col_ind):
            max_scores[row] = max(max_scores[row], scores[row, column])
        return max_scores

    def _compute_f1(self, predicted_bag, gold_bag):
        intersection = len(gold_bag.intersection(predicted_bag))
        if not predicted_bag:
            precision = 1.0
        else:
            precision = intersection / float(len(predicted_bag))
        if not gold_bag:
            recall = 1.0
        else:
            recall = intersection / float(len(gold_bag))
        f1 = (
            (2 * precision * recall) / (precision + recall)
            if not (precision == 0.0 and recall == 0.0)
            else 0.0
        )
        return f1

    def _match_numbers_if_present(self, gold_bag, predicted_bag):
        gold_numbers = set()
        predicted_numbers = set()
        for word in gold_bag:
            if self._is_number(word):
                gold_numbers.add(word)
        for word in predicted_bag:
            if self._is_number(word):
                predicted_numbers.add(word)
        if (not gold_numbers) or gold_numbers.intersection(predicted_numbers):
            return True
        return False

    def _is_number(self, text):
        try:
            float(text)
            return True
        except ValueError:
            return False

    def _remove_articles(self, text):
        return _ARTICLES.sub(" ", text)

    def _white_space_fix(self, text):
        return " ".join(text.split())

    def _remove_punc(self, text):
        exclude = set(string.punctuation)
        if not self._is_number(text):
            return "".join(ch for ch in text if ch not in exclude)
        else:
            return text

    def _fix_number(self, text):
        return str(float(text)) if self._is_number(text) else text

    def _tokenize(self, text):
        return re.split(" |-", text)

    def _normalize(self, answer):
        tokens = [
            self._white_space_fix(self._remove_articles(self._fix_number(self._remove_punc(token.lower()))))
            for token in self._tokenize(answer)
        ]
        tokens = [token for token in tokens if token.strip()]
        normalized = " ".join(tokens).strip()
        return normalized

    def aggregation(self):
        """
        :returns: {str: [float] -> float}
            A dictionary where keys are the names of submetrics and values are
            functions that aggregate a list of metrics
        """
        return {
            "em": mean,
            "f1": mean
        }

    def higher_is_better(self):
        """
        :returns: {str: bool}
            A dictionary where keys are the names of submetrics and values are
            whether a higher value of the submetric is better
        """
        return {
            "em": True,
            "f1": True
        }
