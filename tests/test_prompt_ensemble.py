import numpy as np
import pytest

from esc_robustness.prompt_ensemble import ensemble_embeddings


def test_ensemble_is_fixed_for_all_classes_and_independent_of_batch_size():
    class FakeBackend:
        def __init__(self):
            self.calls = []

        def get_text_embedding(self, prompts, use_tensor=False):
            self.calls.extend(prompts)
            return np.array([[len(p), sum(map(ord, p)) % 71, 1] for p in prompts], dtype=float)

    classes = {0: "dog_bark", 1: "rain"}
    templates = ["The <sound>.", "A recording of <sound>."]
    one, two = FakeBackend(), FakeBackend()
    a, raw = ensemble_embeddings(one, classes, templates, 1)
    b, _ = ensemble_embeddings(two, classes, templates, 2)
    np.testing.assert_allclose(a, b)
    np.testing.assert_allclose(np.linalg.norm(a, axis=1), 1)
    assert raw.shape == (2, 2, 3)
    assert one.calls == two.calls
    assert one.calls == [
        "The dog bark.",
        "A recording of dog bark.",
        "The rain.",
        "A recording of rain.",
    ]
    with pytest.raises(ValueError, match="<sound>"):
        ensemble_embeddings(one, classes, ["missing placeholder"])
