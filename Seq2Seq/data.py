import spacy
from torchtext.legacy.datasets import Multi30k
from torchtext.legacy.data import Field, BucketIterator

class MultiDataset:
    def __init__(self, ):
        super().__init__()
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

        self.SRC = Field(tokenize=self.tokenize_de,
            init_token='<sos>',
            eos_token='<eos>',
            lower=True)

        self.TRG = Field(tokenize=self.tokenize_en,
                    init_token='<sos>',
                    eos_token='<eos>',
                    lower=True)
        
    def tokenize_de(self, text):
        """
        Text to German tokens

        Args:
            text: Text you want to convert to token

        Returns:
            Reverse order of converted tokens
        """
        return [tok.text for tok in self.spacy_de.tokenizer(text)][::-1]

    def tokenize_en(self, text):
        """
        Text to English tokens

        Args:
            text: Text you want to convert to token

        Returns:
            converted tokens        
        """
        return [tok.text for tok in self.spacy_en.tokenizer(text)]
    
    def get_dataset(self):
        """
        Return train, validation, test dataset

        Returns:
            train_data
            valid_data
            test_data
        """
        train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(self.SRC, self.TRG))
        print(f"Number of training examples: {len(train_data.examples)}")
        print(f"Number of validation examples: {len(valid_data.examples)}")
        print(f"Number of testing examples: {len(test_data.examples)}")
        return train_data, valid_data, test_data
    
    def build_vocab(self, dataset, min_freq=2):
        """
        Make vocab dictionary from dataset

        Args:
            dataset: Dataset you want to make vocab dictionary
            min_freq: Minimum number of appearances of words to be made into a dictionary
        """
        self.SRC.build_vocab(dataset, min_freq=min_freq)
        self.TRG.build_vocab(dataset, min_freq=min_freq)
        print(f"Unique tokens in source (de) vocabulary: {len(self.SRC.vocab)}")
        print(f"Unique tokens in target (en) vocabulary: {len(self.TRG.vocab)}")
