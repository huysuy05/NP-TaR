from datasets import load_dataset

from .base_task import BaseClassificationTask

SST2_LABELS = ["positive", "negative"]
SST2_LETTERS = ["A", "B"]
SST2_DECODER_CHOICES = [f" {letter}" for letter in SST2_LETTERS]
SST2_TEMPLATE = (
    "You are a movie critic. Decide whether the review is positive or negative, "
    "and reply with only the letter of your choice.\n\nChoices:\n{choices}\n\nReview:\n{text}\n\nAnswer: "
)


def format_sst2_instruction(text: str) -> str:
    choices_block = "\n".join(
        f"{letter}. {label}" for letter, label in zip(SST2_LETTERS, SST2_LABELS)
    )
    return SST2_TEMPLATE.format(choices=choices_block, text=text)


class SST2Task(BaseClassificationTask):
    def __init__(self, split: str = "validation"):
        ds = load_dataset('glue', 'sst2')
        self.dataset = ds[split]
        # Labels are positive and negative
        self._labels_ = SST2_LABELS

    @property
    def name(self):
        return "sst2"
    
    @property
    def labels(self):
        return self._labels_
    
    def get_test_texts(self):
        return [ex["sentence"] for ex in self.dataset]

    def get_test_labels(self):
        return [int(ex["label"]) for ex in self.dataset]

    def format_with_instruction(self, text: str) -> str:
        return format_sst2_instruction(text)

    @property
    def decoder_choice_texts(self):
        return SST2_DECODER_CHOICES
