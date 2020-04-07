import torch
from transformers import BartTokenizer, BartForConditionalGeneration


class BertSummarizer:
    def __init__(self):
        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer_summarize = BartTokenizer.from_pretrained('bart-large-cnn')
        self.model_summarize = BartForConditionalGeneration.from_pretrained('bart-large-cnn')
        self.model_summarize.to(self.torch_device)
        self.model_summarize.eval()

    def create_summary(self, text: str) -> str:  # this will take time on the first time since it downloads the model
        text_input_ids = self.tokenizer_summarize.batch_encode_plus(
            [text], return_tensors='pt', max_length=1024)['input_ids'].to(self.torch_device)
        summary_ids = self.model_summarize.generate(text_input_ids,
                                                    num_beams=10,
                                                    length_penalty=1.2,
                                                    max_length=1024,
                                                    min_length=64,
                                                    no_repeat_ngram_size=4)
        summary = self.tokenizer_summarize.decode(summary_ids.squeeze(), skip_special_tokens=True)
        return summary
