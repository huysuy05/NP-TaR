class BaseClassificationTask:
    """
    Interface for classification tasks used in NullBench.
    """

    def get_test_texts(self):
        raise NotImplementedError

    def get_test_labels(self):
        raise NotImplementedError

    @property
    def labels(self):
        raise NotImplementedError

    @property
    def name(self):
        raise NotImplementedError

    def format_with_instruction(self, text: str) -> str:
        """Return model-ready text. Default: passthrough (no instruction)."""
        return text

    @property
    def decoder_choice_texts(self):
        labels = list(self.labels) if hasattr(self, "labels") else []
        return [f" {chr(ord('A') + idx)}" for idx, _ in enumerate(labels)]


# Backward-compatible alias (legacy typo)
BaseClassifcationTask = BaseClassificationTask
