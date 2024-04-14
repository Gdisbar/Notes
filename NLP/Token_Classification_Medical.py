# %%

import itertools
from tqdm import tqdm
import numpy as np
import torch
from transformers import BertJapaneseTokenizer, BertForTokenClassification
import pytorch_lightning as pl

# from torch.utils.data import DataLoader
# import from_XML_to_json as XtC
# import random
# import json
# import unicodedata
# import pandas as pd

# %%
# 8-16 
# PyTorch Lightning model
class BertForTokenClassification_pl(pl.LightningModule):
        
    def __init__(self, model_name, num_labels, lr):
        super().__init__()
        self.save_hyperparameters()
        self.bert_tc = BertForTokenClassification.from_pretrained(
            model_name,
            num_labels=num_labels
        )
        
    def training_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        loss = output.loss
        self.log('train_loss', loss)
        return loss
        
    def validation_step(self, batch, batch_idx):
        output = self.bert_tc(**batch)
        val_loss = output.loss
        self.log('val_loss', val_loss)
        
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)



# %%
class NER_tokenizer_BIO(BertJapaneseTokenizer):

    # The number of categories of named entities `num_entity_type` at initialization
    # make it accept.
    def __init__(self, *args, **kwargs):
        self.num_entity_type = kwargs.pop('num_entity_type')
        super().__init__(*args, **kwargs)

    def encode_plus_tagged(self, text, entities, max_length):
        """
        Given a sentence and named entities,
        Encode and create a label string.
        """
        # Divide the text before and after the named entity and label each.
        splitted = [] # Add the string after division
        position = 0
        
        for entity in entities:
            start = entity['span'][0]
            end = entity['span'][1]
            label = entity['type_id']
            splitted.append({'text':text[position:start], 'label':0})
            splitted.append({'text':text[start:end], 'label':label})
            position = end
        splitted.append({'text': text[position:], 'label':0})
        splitted = [ s for s in splitted if s['text'] ]

        # Tokenize and label each segmented sentence
        tokens = [] 
        labels = [] 
        for s in splitted:
            tokens_splitted = self.tokenize(s['text'])
            label = s['label']
            if label > 0: # 固有表現
                # First, assign I-tags to all tokens
                # Number order O-tag: 0, B-tag: 1 ~ number of tags, I-tag: number of tags ~
                labels_splitted =  \
                    [ label + self.num_entity_type ] * len(tokens_splitted)
                # Make the first token a B-tag
                labels_splitted[0] = label
            else: 
                labels_splitted =  [0] * len(tokens_splitted)
            
            tokens.extend(tokens_splitted)
            labels.extend(labels_splitted)

        # Encode it and put it into a format that can be input to BERT.
        input_ids = self.convert_tokens_to_ids(tokens)
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length',
            truncation=True
        ) 

        # Add Special Tokens to Labels
        # Cut by max_length and put labels before and after to add [CLS] and [SEP]
        labels = [0] + labels[:max_length-2] + [0]
        # If it is less than max_length, add the missing part to the end
        labels = labels + [0]*( max_length - len(labels) )
        encoding['labels'] = labels

        return encoding

    def encode_plus_untagged(
        self, text, max_length=None, return_tensors=None
    ):
        """
        Tokenize the sentences and identify the position of each token in the sentence.
        Same as encode_plus_untagged in IO method tokenizer
        """
        # Tokenize the text and associate each token with the character string in the text.
        tokens = [] # Add tokens.
        tokens_original = [] # Add the character strings in the sentence corresponding to the token.
        words = self.word_tokenizer.tokenize(text) # Split into words with MeCab
        for word in words:
            # Split word into subwords
            tokens_word = self.subword_tokenizer.tokenize(word) 
            tokens.extend(tokens_word)
            if tokens_word[0] == '[UNK]': # Dealing with unknown words
                tokens_original.append(word)
            else:
                tokens_original.extend([
                    token.replace('##','') for token in tokens_word
                ])

        # Find the position of each token in the sentence. (considering blank positions)
        position = 0
        spans = [] # Add token positions.
        for token in tokens_original:
            l = len(token)
            while 1:
                if token != text[position:position+l]:
                    position += 1
                else:
                    spans.append([position, position+l])
                    position += l
                    break

        # Encode it and put it into a format that can be input to BERT.
        input_ids = self.convert_tokens_to_ids(tokens) 
        encoding = self.prepare_for_model(
            input_ids, 
            max_length=max_length, 
            padding='max_length' if max_length else False, 
            truncation=True if max_length else False
        )
        sequence_length = len(encoding['input_ids'])
        # Added dummy span for special token [CLS].
        spans = [[-1, -1]] + spans[:sequence_length-2] 
        # Added dummy spans for special tokens [SEP], [PAD].
        spans = spans + [[-1, -1]] * ( sequence_length - len(spans) ) 

        # Make it a torch.Tensor if necessary.
        if return_tensors == 'pt':
            encoding = { k: torch.tensor([v]) for k, v in encoding.items() }

        return encoding, spans

    @staticmethod
    def Viterbi(scores_bert, num_entity_type, penalty=10000):
        """
        Find the optimal solution with the Viterbi algorithm.
        """
        m = 2*num_entity_type + 1
        penalty_matrix = np.zeros([m, m])
        for i in range(m):
            for j in range(1+num_entity_type, m):
                if not ( (i == j) or (i+num_entity_type == j) ): 
                    penalty_matrix[i,j] = penalty
        path = [ [i] for i in range(m) ]
        scores_path = scores_bert[0] - penalty_matrix[0,:]
        scores_bert = scores_bert[1:]

        

        for scores in scores_bert:
            assert len(scores) == 2*num_entity_type + 1
            score_matrix = np.array(scores_path).reshape(-1,1) \
                + np.array(scores).reshape(1,-1) \
                - penalty_matrix
            scores_path = score_matrix.max(axis=0)
            argmax = score_matrix.argmax(axis=0)
            path_new = []
            for i, idx in enumerate(argmax):
                path_new.append( path[idx] + [i] )
            path = path_new

        labels_optimal = path[np.argmax(scores_path)]
        return labels_optimal

    def convert_bert_output_to_entities(self, text, scores, spans):
        """
        Obtain named entities from sentences, classification scores, and the position of each token.
        Classification scores are two-dimensional arrays of size (series length, number of labels)
        """
        assert len(spans) == len(scores)
        num_entity_type = self.num_entity_type
        
        # Remove parts corresponding to special tokens
        scores = [score for score, span in zip(scores, spans) if span[0]!=-1]
        spans = [span for span in spans if span[0]!=-1]
        
        # Determine the predicted value of the label with the Viterbi algorithm.
        labels = self.Viterbi(scores, num_entity_type)

        # Tokens with the same label are grouped together to extract named entities.
        entities = []
        for label, group in itertools.groupby(enumerate(labels), key=lambda x: x[1]):
            group = list(group)
            start = spans[group[0][0]][0]
            end = spans[group[-1][0]][1]

            if label != 0: # if it is a named entity
                if 1 <= label <= num_entity_type:
                     # Add new entity if label is `B-`
                    entity = {
                        "name": text[start:end],
                        "span": [start, end],
                        "type_id": label
                    }
                    entities.append(entity)
                else:
                    # If the label is `I-`, update the last entity
                    entity['span'][1] = end 
                    entity['name'] = text[entity['span'][0]:entity['span'][1]]
                
        return entities

