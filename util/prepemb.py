import pandas as pd
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoTokenizer, AutoModel

class SbertComparer:
    def __init__(self, df, df_imdrf, model):
        self.model = model
        self.df = df
        self.df_imdrf = df_imdrf

    def extractbestmatch(self, emb1, emb2, level):
        '''Function that aims to prepare the embeddings and compare them to find the best match.
        :param emb1: Embeddings for the first text.
        :param emb2: Embeddings for the class text.
        :param typetext: Type of the text column. (Keywords, Cleaned, FOI_TEXT, first2sents)
        :param typelabel: Type of the label column. (multiple_terms, single_term, Definition)
        :param df: DataFrame containing the text columns.
        :param level: Level of the class column. (IMDRF, CatOneIMDRF, CatTwoIMDRF)
        '''
        # Comparison of the embeddings
        best_match, max_sim = self.computesim(emb1, emb2)

        original_code = self.df[level].tolist()
        orig_desc = self.codetodesk(original_code)
        pred_desc = self.codetodesk(best_match)

        result_df = pd.DataFrame({'Original text': self.df['FOI_TEXT'], 'Original IMDRF Description': orig_desc, 'Original IMDRF Code': self.df[level],'Predicted IMDRF Description': self.df_imdrf['multiple_terms'], 'Predicted IMDRF Code': best_match, 'Predicted IMDRF Description': pred_desc, 'Similarity Score': max_sim})
        result_df['Correct Prediction'] = result_df.apply(lambda x: self.check_prediction(x['Original IMDRF Code'], x['Predicted IMDRF Code'], level), axis=1)
        print(result_df['Correct Prediction'].value_counts())
        return result_df
    
    def codetodesk(self, code_list):
        '''Function that returns a list with the descriptions of the IMDRF code.
        :param code_list: list of IMDRF codes.
        :param desc_list: list containing the descriptions of the codes.
        '''
        desc_list = []

        for code in code_list:
            desc_list.append(self.df_imdrf[self.df_imdrf['IMDRF Code'] == code]['multiple_terms'].values[0])
        return desc_list


    def computesim(self, emb1, emb2):
        best_match = []
        max_sim = []

        for text_embedding in emb1:
            similarities = self.model.similarity(text_embedding, emb2).squeeze()
            max_index = int(similarities.argmax())
            max_similarity = similarities[max_index].item()
            best_match.append(self.df_imdrf['IMDRF Code'].iloc[max_index])
            max_sim.append(max_similarity)
        return best_match, max_sim

    def check_prediction(self, real_codes, predicted_code, level):
        if level == 'IMDRF':
            codes = real_codes.split(', ')
            return 1 if predicted_code in codes else 0
        
        elif level == 'CatOneIMDRF':
            short_pred = predicted_code[:3]
            codes = [code[:3] for code in real_codes.split(', ')]
            return 1 if short_pred in codes else 0 
        

class BCBComparer:
    def __init__(self, df, df_imdrf, tokenizer, model):
        self.df = df
        self.df_imdrf = df_imdrf
        self.tokenizer = tokenizer
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    def weighted_avg(self, keylist, simlist, type):
        emb = []
        for key in keylist:
            encoded_input = self.tokenizer(key, padding=True, truncation=True, max_length=512, return_tensors='pt')
            encoded_input = {'input_ids': encoded_input['input_ids'].to(self.device), 'attention_mask': encoded_input['attention_mask'].to(self.device)}

            with torch.no_grad():
                model_output = self.model(input_ids=encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
                if type == 'cls':
                    embeddings = model_output.last_hidden_state[:, 0, :]
                elif type == 'mean':
                    embeddings = torch.mean(model_output.last_hidden_state, dim=1)
            emb.append(embeddings.cpu().numpy())

        emb = np.array(emb)
        simlist = np.array(simlist)
        avg = np.average(emb, axis=0, weights=simlist)
        return avg
    
    def compute_bioclinical_embeddings(self, column, type):
        encoded_input = self.tokenizer(column.tolist(), padding=True, truncation=True, max_length=64, return_tensors='pt')
        encoded_input = {
            'input_ids': encoded_input['input_ids'].to(self.device),
            'attention_mask': encoded_input['attention_mask'].to(self.device)
        }

        with torch.no_grad():
            model_output = self.model(input_ids=encoded_input['input_ids'], attention_mask=encoded_input['attention_mask'])
            if type == 'cls':
                embeddings = model_output.last_hidden_state[:, 0, :]
            elif type == 'mean':
                embeddings = torch.mean(model_output.last_hidden_state, dim=1)

        return embeddings.cpu().numpy()
    
    def compare_embeddings(self, embed1, embed2):
        cos_sim = cosine_similarity(embed1.reshape(1, -1), embed2)
        max_index = np.argmax(cos_sim)
        max_similarity = cos_sim[0, max_index]
        best_match = self.df_imdrf['IMDRF Code'].iloc[max_index]
        return best_match, max_similarity
    
    def check_prediction(self, real_codes, predicted_code, level):
        if isinstance(real_codes, pd.Series):
            real_codes = real_codes.iloc[0]  # Assicurati che real_codes sia una stringa

        real_codes = str(real_codes) if not isinstance(real_codes, str) else real_codes
        
        if level == 'IMDRF':
            codes = real_codes.split(', ')
            return 1 if predicted_code in codes else 0
        
        elif level == 'CatOneIMDRF':
            short_pred = predicted_code[:3]
            codes = [code[:3] for code in real_codes.split(', ')]
            return 1 if short_pred in codes else 0

    def extractbestfit(self, embtext, emblabel, level):
        best_match, max_sim = zip(*[self.compare_embeddings(text, emblabel) for text in embtext])
        
        result_df = pd.DataFrame({'Original IMDRF': self.df[level], 'Predicted IMDRF': best_match, 'Similarity Score': max_sim})
        result_df['Correct Prediction'] = result_df.apply(lambda x: self.check_prediction(x['Original IMDRF'], x['Predicted IMDRF'], level), axis=1)
        print(result_df['Correct Prediction'].value_counts())

        return result_df