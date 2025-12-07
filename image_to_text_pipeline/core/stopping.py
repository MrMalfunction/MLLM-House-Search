"""
Stopping criteria for generation control.
Stops generation when specific text sequences appear or excessive repetition detected.
"""

from transformers import StoppingCriteria


class TextStoppingCriteria(StoppingCriteria):
    """Stop generation when specific text sequences appear in the output"""

    def __init__(self, tokenizer, stop_sequences, initial_length):
        self.tokenizer = tokenizer
        self.stop_sequences = stop_sequences
        self.initial_length = initial_length

    def __call__(self, input_ids, scores, **kwargs):  # type: ignore
        # Decode only newly generated tokens (skip special tokens to match content)
        generated_ids = input_ids[0][self.initial_length :]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Check if any stop sequence appears
        for stop_seq in self.stop_sequences:
            if stop_seq in generated_text:
                return True

        # Also stop if we see excessive repetition (sign of model breakdown)
        if len(generated_text) > 500:
            words = generated_text[-500:].split()
            if len(words) > 50:
                # Check if last 50 words are highly repetitive
                unique_ratio = len(set(words[-50:])) / 50.0
                if unique_ratio < 0.3:  # Less than 30% unique words
                    return True

        return False
