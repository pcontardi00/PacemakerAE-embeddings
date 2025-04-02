import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer, AutoModel

class ResultMatrixSTS:
    def __init__(self, df, df_imdrf, level):
        """
        Initialize the class with dataframes and level.
        """
        self.df = df
        self.df_imdrf = df_imdrf
        self.level = level

    def printresultmatrix(self, model, tokenizer=None, strategy=None):
        """
        Prints table with the results of the embedding comparison.
        
        Args:
            model: The embedding model (SBERT, BERT, or DeepSeek).
            tokenizer: Tokenizer for BERT models (optional for SBERT/DeepSeek).
            strategy: Strategy for BERT embeddings ('cls', 'mean', or None for SBERT/DeepSeek).
        """
        sum_correct = []

        # Set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        if hasattr(model, 'to'):  # Move model to device if it's a PyTorch model
            model.to(device)

        # Compute embeddings for text columns
        text_columns = [
            'Cleaned', 'NoPunctuation', 'FOI_TEXT', 'f2sent', 'trimsent',
            'cleanf2', 'cleantrim', 'nopunf2', 'nopuntrim'
        ]
        text_embeddings = [self.compute_embeddings(model, tokenizer, self.df[col], strategy, device) for col in text_columns]

        # Compute embeddings for label columns
        label_columns = ['single_term', 'multiple_terms', 'Definition']
        label_embeddings = [self.compute_embeddings(model, tokenizer, self.df_imdrf[col], strategy, device) for col in label_columns]

        # Compare embeddings and calculate accuracy
        for text_emb in text_embeddings:
            for label_emb in label_embeddings:
                sum_correct.append(self.extractbestmatch(text_emb, label_emb))

        # Create and populate the output dataframe
        index_df = ['single term', 'multiple terms', 'definition']
        columns_df = [
            'original text', 'clean text', 'no punctuation', 'trim text',
            'first two sentences', 'clean+trim', 'clean+f2', 'nopun+trim', 'nopun+f2'
        ]
        output = pd.DataFrame(np.array(sum_correct).reshape(len(columns_df), len(index_df)).T,
                              index=index_df, columns=columns_df)

        print(output)
        return output

    def compute_embeddings(self, model, tokenizer, column, strategy, device):
        """
        Compute embeddings for a given column based on the model and strategy.
        """
        if strategy is None:  # SBERT or DeepSeek
            return model.encode(column.tolist())
        else:  # BERT
            return self.compute_bert_embeddings(model, tokenizer, column, strategy, device)

    def compute_bert_embeddings(self, model, tokenizer, column, type, device):
        """
        Compute BERT embeddings using CLS or mean pooling.
        """
        encoded_input = tokenizer(column.tolist(), padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = model(**encoded_input)
            if type == 'cls':
                embeddings = model_output.last_hidden_state[:, 0, :]
            elif type == 'mean':
                embeddings = torch.mean(model_output.last_hidden_state, dim=1)

        return embeddings.cpu().numpy()

    def extractbestmatch(self, emb1, emb2):
        """
        Find the best match for each embedding in emb1 from emb2.
        """
        best_match = self.computebestmatch(emb1, emb2)
        original_code = self.df[self.level].tolist()
        correct_score = self.check_prediction(original_code, best_match)
        return sum(correct_score) / len(correct_score)

    def computebestmatch(self, emb1, emb2):
        """
        Compute the best match using cosine similarity.
        """
        best_match = []
        for text_embedding in emb1:
            similarities = cosine_similarity([text_embedding], emb2).squeeze()
            max_index = int(similarities.argmax())
            best_match.append(self.df_imdrf['IMDRF Code'].iloc[max_index])
        return best_match

    def check_prediction(self, real_codes, predicted_codes):
        """
        Check if each predicted code matches the corresponding real code.
        """
        scores = []
        for real_code, predicted_code in zip(real_codes, predicted_codes):
            if self.level == 'IMDRF':
                scores.append(1 if predicted_code == real_code else 0)
            elif self.level == 'CatOneIMDRF':
                short_pred = predicted_code[:3]
                short_real = real_code[:3]
                scores.append(1 if short_pred == short_real else 0)
        return scores