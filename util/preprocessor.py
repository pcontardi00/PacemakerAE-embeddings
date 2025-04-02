import re
import pandas as pd
from nltk.tokenize import sent_tokenize
from transformers import AutoTokenizer

model_name = "sentence-transformers/all-MiniLM-L6-v2"
tokenizer_bert = AutoTokenizer.from_pretrained(model_name)

class TextPreprocessor:
    def __init__(self, limit=512):
        """
        Initialize the TextPreprocessor class.
        :param limit: Maximum token limit for the trimtextlen function.
        """
        self.limit = limit

    def clean_text(self, text, punct_flag=False):
        """Cleans the text using regex (lowercasing, removing digits, etc.)."""
        if pd.isna(text):
            return ''
        else:
            text = text.lower()
            text = re.sub(r'\(.*?\)', '', text)  # removes text inside parentheses
            text = re.sub(r'\d+', '', text)  # removes digits
            text = text.replace("�", " ")  # removes invalid replacement characters
            if punct_flag:
                text = re.sub(r'[^\w\s]', '', text)  # remove punctuation
            text = re.sub(r'\s+', ' ', text)  # remove extra spaces
            return text.strip()
        
    def remove_punct(self, text):
        """Removes punctuation from the text."""
        if pd.isna(text):
            return ''
        else:
            text = re.sub(r'[^\w\s]', '', text)
            return text.strip()
        
    def remove_stopwords(self, text, stoplist):
        '''Remove stopwords contained in stoplist from text'''
        if pd.isna(text):
            return ''
        else:
            text = ' '.join([word for word in text.split() if word not in stoplist])
            return text

    def first_two_sentences(self, text):
        """Returns the first two sentences of the text, cleaned."""
        if pd.isna(text):
            return ''
        else:
            sentences = sent_tokenize(text)
            if len(sentences) < 2:
                text = self.clean_text(text)
                return text
            else:
                # Clean the first two sentences
                first_two = ' '.join(sentences[:2])
                return first_two

    def remredclass(self, sample):
        """
        Removes redundant classes from a comma-separated string of class codes.
        :param sample: Input string with class codes separated by commas.
        :return: String with redundant classes removed.
        """
        if pd.isna(sample):
            return None
        class_list = sample.split(', ')
        contained_codes = set()

        if len(class_list) == 1:
            return sample
        else:
            for i, code1 in enumerate(class_list):
                for j, code2 in enumerate(class_list):
                    if i != j and code1 in code2:
                        contained_codes.add(code1)

        filt_class = [code for code in class_list if code not in contained_codes]
        return ', '.join(filt_class)

    def extract_category(self, x):
        """
        Extracts hierarchical categories from a string of problem codes.
        Generates two levels of categories: CatOneIMDRF and CatTwoIMDRF.
        :param x: Input string with problem codes.
        :return: Tuple with two strings (CatOneIMDRF, CatTwoIMDRF).
        """
        if pd.isna(x):
            return None, None
        else:
            list_cat = x.split(', ')
            cat_one = []
            cat_two = []

            for cat in list_cat:
                first_cat = cat[0:3]  # First hierarchical level
                cat_one.append(first_cat)
                if len(cat) > 3:
                    second_cat = cat[0:5]  # Second hierarchical level
                    cat_two.append(second_cat)
            
            return ', '.join(set(cat_one)), ', '.join(set(cat_two))
        
    def trimlastcompBERT(self, text, tokenizer=tokenizer_bert, token_limit=512):
        sentences = sent_tokenize(text)  # Split the text into sentences
        trimmed_text = ""
        total_tokens = 0

        for sentence in sentences:
            tokenized_out = tokenizer.tokenize(sentence, truncation=False, padding=False, return_tensors=None)
            numtokens = len(tokenized_out)
            
            if total_tokens + numtokens > token_limit:
                break

            trimmed_text += sentence + " "
        
        return trimmed_text.strip()