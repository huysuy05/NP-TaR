from datasets import load_dataset
from .base_task import BaseClassificationTask

AG_NEWS_LABELS = ["world", "sci/tech", "business", "sports"]
LETTER_CHOICES = ["A", "B", "C", "D"]
DECODER_CHOICE_TEXTS = [f" {letter}" for letter in LETTER_CHOICES]
CHOICES_BLOCK = "\n".join(f"{letter}. {label}" for letter, label in zip(LETTER_CHOICES, AG_NEWS_LABELS))
INSTRUCTION_TEMPLATE = (
    "You are an expert news editor. Given the following article, choose the "
    "best matching category by replying with only the letter of your choice.\n\n"
    f"Choices:\n{CHOICES_BLOCK}\n\n"
    "Article:\n{text}\n\nAnswer: "
)


def format_ag_news_instruction(text: str) -> str:
    return INSTRUCTION_TEMPLATE.format(text=text)


def label_to_choice_text(label_idx: int) -> str:
    return DECODER_CHOICE_TEXTS[label_idx]


class AGNewsTask(BaseClassificationTask):
    """Task for AG News dataset inherited by BaseClassificationTask."""

    def __init__(self, split: str = "test"):
        self.dataset = load_dataset("ag_news")[split]
        self._labels_ = AG_NEWS_LABELS

    @property
    def name(self):
        return "ag_news"
    
    @property
    def labels(self):
        return self._labels_
    
    def get_test_texts(self):
        return [ex["text"] for ex in self.dataset]

    def get_test_labels(self):
        # labels are ints 0..3
        return [ex["label"] for ex in self.dataset]

    def format_with_instruction(self, text: str) -> str:
        return format_ag_news_instruction(text)

    @property
    def letter_choices(self):
        return LETTER_CHOICES

    @property
    def decoder_choice_texts(self):
        return DECODER_CHOICE_TEXTS


