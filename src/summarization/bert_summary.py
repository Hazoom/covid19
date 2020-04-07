import torch
from transformers import BartTokenizer, BartForConditionalGeneration


class BertSummarizer:
    def __init__(self):
        # This will take time on the first time since it downloads the model

        self.torch_device = 'cuda' if torch.cuda.is_available() else 'cpu'
        model_name = 'bart-large-cnn'
        print(f'Initializing BartTokenizer with model: {model_name} ...')
        self.tokenizer_summarize = BartTokenizer.from_pretrained(model_name)
        print(f'Finished initializing BartTokenizer with model: {model_name}')

        print(f'Initializing BartForConditionalGeneration with model: {model_name} ...')
        self.model_summarize = BartForConditionalGeneration.from_pretrained(model_name)
        print(f'Finished initializing BartForConditionalGeneration with model: {model_name}')
        self.model_summarize.to(self.torch_device)
        self.model_summarize.eval()

    def create_summary(self, text: str,
                       repetition_penalty=1.0) -> str:
        text_input_ids = self.tokenizer_summarize.batch_encode_plus(
            [text], return_tensors='pt', max_length=1024)['input_ids'].to(self.torch_device)
        summary_ids = self.model_summarize.generate(text_input_ids,
                                                    num_beams=10,
                                                    length_penalty=1.2,
                                                    max_length=1024,
                                                    min_length=64,
                                                    no_repeat_ngram_size=4,
                                                    repetition_penalty=repetition_penalty)
        summary = self.tokenizer_summarize.decode(summary_ids.squeeze(), skip_special_tokens=True)
        return summary
