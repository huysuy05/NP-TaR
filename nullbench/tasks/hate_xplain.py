from datasets import load_dataset

from .base_task import BaseClassificationTask

HATEXPLAIN_LABELS = ["normal", "offensive", "hate"]
HATEXPLAIN_LETTERS = ["A", "B", "C"]
HATEXPLAIN_DECODER_CHOICES = [f" {letter}" for letter in HATEXPLAIN_LETTERS]
HATEXPLAIN_TEMPLATE = (
    "You are moderating an online forum. Read the post and classify it by replying "
    "with only the letter that best matches the tone.\n\nChoices:\n{choices}\n\nPost:\n{text}\n\nAnswer: "
)


def format_hatexplain_instruction(text: str) -> str:
    choices_block = "\n".join(
        f"{letter}. {label}" for letter, label in zip(HATEXPLAIN_LETTERS, HATEXPLAIN_LABELS)
    )
    return HATEXPLAIN_TEMPLATE.format(choices=choices_block, text=text)


class HateXplainTask(BaseClassificationTask):
    """
    HateXplain dataset.
    3 classes:
        0 = normal
        1 = offensive
        2 = hate speech
    """

    def __init__(self, split: str = "test"):
        ds = load_dataset("hatexplain")
        # only has train/validation/test
        self.dataset = ds[split]

        self._labels_ = HATEXPLAIN_LABELS

    @property
    def name(self):
        return "hatexplain"

    @property
    def labels(self):
        return self._labels_

    def _label_to_int(self, label_dict):
        """
        HateXplain stores labels as {"label": {"some": value}}
        Each example has multiple annotators → majority vote.
        """
        label_counts = {"normal": 0, "offensive": 0, "hate": 0}
        for annotator_label in label_dict:
            if annotator_label in label_counts:
                label_counts[annotator_label] += 1

        # majority vote
        majority = max(label_counts, key=label_counts.get)

        return {
            "normal": 0,
            "offensive": 1,
            "hate": 2
        }[majority]

    def get_test_texts(self):
        # join tokenized post text into a single string
        return [" ".join(ex["post_tokens"]) for ex in self.dataset]

    def get_test_labels(self):
        return [self._label_to_int(ex["annotators"]["label"]) for ex in self.dataset]

    def format_with_instruction(self, text: str) -> str:
        return format_hatexplain_instruction(text)

    @property
    def decoder_choice_texts(self):
        return HATEXPLAIN_DECODER_CHOICES
