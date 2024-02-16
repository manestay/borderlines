# credits: https://github.com/bigscience-workshop/t-zer

# make this as a file named rank_util.py

import logging
from typing import Optional

import torch
from torch import nn
from transformers import AutoModelForSeq2SeqLM, AutoModelForCausalLM, MODEL_FOR_CAUSAL_LM_MAPPING, \
    MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING
from dataclasses import dataclass
from itertools import chain
from typing import Optional, Union

import torch
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy



logger = logging.getLogger(__name__)

class ModelBase(nn.Module):
    def forward(self, batch):
        raise NotImplementedError

    @staticmethod
    def from_config(config, **kwargs) -> "ModelBase":
        task_mapping = [
            (MODEL_FOR_CAUSAL_LM_MAPPING, DecoderModel),
            (MODEL_FOR_SEQ_TO_SEQ_CAUSAL_LM_MAPPING, EncoderDecoderModel),
        ]
        config_name = config.__class__
        for transformer_model_mapping, model in task_mapping:
            transformer_model_name = transformer_model_mapping.get(config_name, None)
            if transformer_model_name is not None:
                return model(config=config, **kwargs)

        raise NotImplementedError

class EncoderDecoderModel(ModelBase):
    def __init__(self, config, model_name_or_path: Optional[str], parallelize: bool, **kwargs):
        """
        Args:
            config:
            model_name_or_path:
            parallelize:
            device: if parallelize = False, then we use specified device.
        """
        super(EncoderDecoderModel, self).__init__()
        logger.info("Building EncoderDecoderModel")
        if model_name_or_path:
            self._model = AutoModelForSeq2SeqLM.from_pretrained(
                model_name_or_path,
                from_tf=bool(".ckpt" in model_name_or_path),
                config=config,
                device_map='auto',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            logger.info("Training new model from scratch")
            self._model = AutoModelForSeq2SeqLM.from_config(config)

        if parallelize:
            assert torch.cuda.is_available(), "You need at least 1 GPU to call `parallelize` (even though if there is only 1 GPU, there won't be any model parallelism)."
            self._model.parallelize()


    def forward(self, batch) -> torch.Tensor:
        model_inputs = {
            k: batch[k]
            for k in ["input_ids", "attention_mask", "labels"]
        }
        logits = self._model(**model_inputs).logits
        masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
        seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0),
                                         -1)  # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
        predictions = seq_log_prob.argmax(dim=-1)
        return predictions, seq_log_prob

class DecoderModel(ModelBase):
    def __init__(self, config, model_name_or_path: Optional[str], **kwargs):
        super(DecoderModel, self).__init__()
        logger.info("Building DecoderModel")
        if model_name_or_path:
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name_or_path,
                config=config,
                device_map='auto',
                torch_dtype=torch.float16,
                low_cpu_mem_usage=True
            )
        else:
            logger.info("Training new model from scratch")
            self._model = AutoModelForCausalLM.from_config(config)

    def forward(self, batch):
        _, prefix_length = batch["input_ids"].shape
        model_inputs = {
            "input_ids": torch.cat([batch["input_ids"], batch["labels"]], dim=-1),
            "attention_mask": torch.cat([batch["attention_mask"], batch["labels_attention_mask"]], dim=-1),
        }
        # Set position ids correctly to take care of padding tokens between inputs_ids and labels
        # Empty attention_mask is a forbidden value, ie full of zeros. In fact the first element should be 1 as the input
        #   cannot be empty
        assert torch.all(model_inputs["attention_mask"][:,0] == 1), "First element in the attention mask should be 1."
        position_ids = torch.cumsum(model_inputs["attention_mask"].to(torch.long), dim=-1) - 1
        model_inputs["position_ids"] = position_ids

        logits = self._model(**model_inputs).logits[:, prefix_length-1:-1]
        masked_log_probs = batch["labels_attention_mask"].unsqueeze(-1) * torch.log_softmax(logits, dim=-1)
        seq_token_log_probs = torch.gather(masked_log_probs, -1, batch["labels"].unsqueeze(-1))
        seq_log_prob = seq_token_log_probs.squeeze(dim=-1).sum(dim=-1)
        seq_log_prob = seq_log_prob.view(batch["targets"].size(0),
                                         -1)  # TODO(Victor): this reshapes works based on the assumption that all examples have the same number of choices. the pre-processing doesn't make this assumption.
        predictions = seq_log_prob.argmax(dim=-1)
        import pdb; pdb.set_trace()
        return predictions, seq_log_prob




@dataclass
class DataCollatorForMultipleChoice:
    """
    Data collator that will dynamically pad the inputs for multiple choice received.
    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
            sequence if provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
            maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
            different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
            Note that it's very NOT recommended to use fp16 to do any time of inference with T0 as the predictions will vastly differ from the predictions using fp32.
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features):
        num_choices = len(features[0]["input_ids"][0])
        # print(features[0]["input_ids"][0])
        # print(f"num_choices={num_choices}")
        flattened_features = [
            [
                {
                    k: v[0][i]
                    for k, v in feature.items()
                    if k != "targets"
                }
                for i in range(num_choices)
            ]
            for feature in features
        ]
        flattened_features = list(chain(*flattened_features))

        batch = self.tokenizer.pad(
            flattened_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
        )

        # Pad the labels because it's not padded automatically
        max_label_length = max([len(elem["labels"]) for elem in flattened_features])
        batch["labels"] = [
            l + [self.tokenizer.pad_token_id]*(max_label_length - len(l))
            for l in [elem["labels"] for elem in flattened_features]
        ]
        batch["labels_attention_mask"] = [
            m + [0]*(max_label_length - len(m))
            for m in [elem["labels_attention_mask"] for elem in flattened_features]
        ]

        # for k, v in batch.items():
        #     print(k)
        #     print(len(v))
        #     print(v)
        #     v = torch.tensor(v)
        #     print("----")

        # Convert to tensors
        batch = {
            k: torch.tensor(v)
            for k, v in batch.items()
        }

        batch["targets"] = torch.tensor([f.pop("targets") for f in features])
        return batch



def convert_features(tokenizer, data):
    input_texts = data["input_texts"]
    answer_choices_texts = data["answer_choices_texts"]
    target_texts = data["target_texts"]
    bs = len(input_texts)
    padding = "max_length"
    max_length = 64

    tokenized_inputs = tokenizer(
                input_texts,
                padding=padding,
                max_length=max_length,
                truncation=True,
                add_special_tokens=False,
            )

    tokenized_targets = [
        tokenizer(
            options,
            # padding is on the right here.
            padding=padding,
            max_length=max_length,
            truncation=True,
        )
        for options in answer_choices_texts
    ]


    features = {
                k: [
                    [elem for _ in range(len(tokenized_targets[idx]["input_ids"]))]
                    for idx, elem in enumerate(v)
                ]
                for k, v in tokenized_inputs.items()
            }

    features["labels"] = [
        tokenized_targets[idx]["input_ids"]
        for idx in range(bs)
    ]
    features["labels_attention_mask"] = [
        tokenized_targets[idx]["attention_mask"]
        for idx in range(bs)
    ]
    features["targets"] = [
        answer_choices_texts[idx].index(t) if t in answer_choices_texts[idx]  else -1
        for idx, t in enumerate(target_texts)
    ]
    return features
