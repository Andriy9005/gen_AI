from nemo.collections.tts.torch.tts_tokenizers import BaseCharsTokenizer
from nemo.collections.common.tokenizers.text_to_speech.tokenizer_utils import any_locale_text_preprocessing



def lowercase_text_preprocessing(text):
    text = any_locale_text_preprocessing(text)
    text = text.lower()
    return text



class CharsTokenizer(BaseCharsTokenizer):
    PUNCT_LIST = BaseCharsTokenizer.PUNCT_LIST+('+',"—")

    def __init__(
        self,
        chars,
        punct=True,
        apostrophe=True,
        add_blank_at=None,
        pad_with_space=False,
        non_default_punct_list=None,
        text_preprocessing_func=lowercase_text_preprocessing,
    ):
        """Char-based tokenizer.
        Args:
            chars: string that represents all possible characters.
            punct: Whether to reserve grapheme for basic punctuation or not.
            apostrophe: Whether to use apostrophe or not.
            add_blank_at: Add blank to labels in the specified order ("last") or after tokens (any non None),
             if None then no blank in labels.
            pad_with_space: Whether to pad text with spaces at the beginning and at the end or not.
            non_default_punct_list: List of punctuation marks which will be used instead default.
            text_preprocessing_func: Text preprocessing function for correct execution of the tokenizer.
        """
        super().__init__(
            chars=chars,
            punct=punct,
            apostrophe=apostrophe,
            add_blank_at=add_blank_at,
            pad_with_space=pad_with_space,
            non_default_punct_list=non_default_punct_list,
            text_preprocessing_func=text_preprocessing_func,
        )