import re
import numpy as np
import spacy
import logging
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import torch

# the following code has taken inspiration from the following sources:
# https://github.com/luozhouyang/embedrank/blob/master/embedrank/embedrank_d2v.py

nlp = spacy.load('en_core_web_sm')

class EmbedRank(object):

    def __init__(self):
        pass

    def getKWembedding(self, document, weight_flag, model, tokenizer=None, strategy=None, N=None):
        '''
        This function receives the document, a flag value.
        It extracts the keywords from the document and computes the embedding of those.
        - document: the text from which to extract the keywords.
        - weight_flag: a boolean value that indicates if the keywords should be weighted by their similarity scores. If True, the keywords are weighted by their similarity scores. If False, the keywords are not weighted.
        - model: the SentenceTransformer or Transformer model to compute the embeddings.
        - tokenizer: the tokenizer to preprocess the text.
        - strategy: the strategy to compute the embeddings. It can be 'cls' or 'mean'. If None, the embeddings are computed using the model.encode() method.
        - N: the number of keywords to extract. If None, the default value is 3.
        '''
        self.model = model
        self.tokenizer = tokenizer if tokenizer is not None else None
        self.strategy = strategy if strategy is not None else None
        self.N = 3 if N is None else N
        self.weight_flag = weight_flag

        keylist, simlist = self._mmr(document)
        return self.finalembedding(keylist, simlist)

    def _mmr(self, document, _lambda=0.5, nounadjFlag=True):
        '''
        This function receives the document, the model, the tokenizer, the strategy, the lambda, the number of keywords to extract, and a flag to include only nouns and adjectives.
        It returns the selected keywords and their similarity scores.
        '''

        keywords = self.candidate_extraction(document, nounadjFlag)
        if len(keywords) == 0:
            logging.warning('No keywords found in the document')
            return [], [], []
        keywords = self.dropping_contained_kw(keywords) # cleaning the duplicated keywords

        document_embedding = self.obtainembeddings([document])
        keyword_embeddings = self.obtainembeddings(keywords)

        # Similarity scores
        keyword_doc_similarities = cosine_similarity(keyword_embeddings, document_embedding.reshape(1, -1))
        keyword_keyword_similarities = cosine_similarity(keyword_embeddings)

        # 1st iteration
        not_selected = list(range(len(keywords)))
        if len(not_selected) == 0:
            logging.warning('Not selected is empty 1')
            return [], [], []
        
        selected_idx = np.argmax(keyword_doc_similarities)
        selected = [selected_idx]
        not_selected.remove(selected_idx)
        selected_mmr_scores = []

        # sel_kw = [keywords[i] for i in selected] to be deleted

        # other iterations
        for _ in range(min(self.N-1, len(not_selected))):
            mmr_distance_to_doc = keyword_doc_similarities[not_selected, :]
            mmr_distance_between_keywords = np.max(keyword_keyword_similarities[not_selected][:, selected], axis=1)

            mmr = _lambda * mmr_distance_to_doc - (1 - _lambda) * mmr_distance_between_keywords.reshape(-1, 1)
            mmr_idx = not_selected[np.argmax(mmr)]
            sel_mmr = mmr[np.argmax(mmr)]

            selected.append(mmr_idx)
            selected_mmr_scores.append(float(sel_mmr[0]))
            not_selected.remove(mmr_idx)
            mmr = np.delete(mmr, np.argmax(mmr))
            # sel_kw.append(keywords[mmr_idx]) to be deleted

        sel_kw = [keywords[i] for i in selected]
        sel_sim = [keyword_doc_similarities[i][0] for i in selected]

        if sel_kw == []:
            logging.warning('No keywords found in the document')
            return [], [], []

        return sel_kw, sel_sim



    def candidate_extraction(self, text, nounadjFlag):
        doc = nlp(text)

        candidate_keywords = []

        for sent in doc.sents:
            current_phrase = []
            for token in sent:
                if nounadjFlag:
                    # Include nouns, adjectives, past tense verbs, and gerunds
                    if token.pos_ in {'NOUN', 'ADJ'} or \
                    (token.pos_ == 'VERB' and (token.morph.get("Tense") == ["Past"] or token.morph.get("VerbForm") == ["Ger"])):
                        current_phrase.append(token.text)
                        
                        # Append the phrase if the token is a noun or a verb meeting the criteria
                        if token.pos_ == 'NOUN' or (token.pos_ == 'VERB' and (token.morph.get("Tense") == ["Past"] or token.morph.get("VerbForm") == ["Ger"])):
                            candidate_keywords.append(' '.join(current_phrase))
                    else:
                        current_phrase = []
                else:
                    # Include only past tense verbs and gerunds
                    if token.pos_ == 'VERB' and (token.morph.get("Tense") == ["Past"] or token.morph.get("VerbForm") == ["Ger"]):
                        current_phrase.append(token.text)
                        candidate_keywords.append(' '.join(current_phrase))
                    else:
                        current_phrase = []

        return candidate_keywords
    

    def dropping_contained_kw(self, kw_list):
        '''This function receives the list of all the keywords extracted and considered as candidates.
        These keywords are going to get filtered in order to remove the ones that are substrings of another.
        '''
        kw_list = list(set(kw_list))
        kw_list = sorted(kw_list, key=len, reverse=False)

        for i in kw_list:
            for j in kw_list:
                if i in j and i != j:
                    kw_list.remove(i)
                    break

        return kw_list
    
    def obtainembeddings(self, document):
        '''
        This function receives the document, the model, the tokenizer, and the strategy to compute the embeddings.
        It returns the embeddings of the document.
        '''
        if self.strategy is None:
            return self.model.encode(document)
        else:
            return self.compute_bert_embeddings(document)
        
    def compute_bert_embeddings(self, column):
        """
        Compute BERT embeddings using CLS or mean pooling.
        """
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        encoded_input = self.tokenizer(column, padding=True, truncation=True, max_length=512, return_tensors='pt')
        encoded_input = {k: v.to(device) for k, v in encoded_input.items()}

        with torch.no_grad():
            model_output = self.model(**encoded_input)
            if self.strategy == 'cls':
                embeddings = model_output.last_hidden_state[:, 0, :]
            elif self.strategy == 'mean':
                embeddings = torch.mean(model_output.last_hidden_state, dim=1)

        return embeddings.cpu().numpy()
    
    def finalembedding(self, keylist, simlist):
        '''
        Takes as input the list of keywords and the similarity scores and returns the final embedding.
        There are two possible ways to compute the final embedding:
        - The average of the embeddings of the keywords, weighted by the similarity scores.
        - The average of the embeddings of the keywords.
        '''
        emb = []
        for key in keylist:
            emb.append(self.obtainembeddings([key]))

        emb = np.array(emb)
        simlist = np.array(simlist)  # Ensure simlist is a numpy array
        # computing the weighted average
        if self.weight_flag:
            return np.average(emb, axis=0, weights=simlist)
        else:
            return np.mean(emb, axis=0)
        
