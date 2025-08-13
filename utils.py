import math
import numpy as np
import random
import torch


class DependencyParsing():
    def __init__(self):
        import spacy
        self.nlp = spacy.load("en_core_web_sm")
        # if it reports cannot find "en_core_web_sm", try to load as below:
        # self.nlp = spacy.load("/usr/local/lib/python3.10/site-packages/en_core_web_sm/en_core_web_sm-3.8.0") # the path is an example

    def parse(self, text, never_empty):
        doc = self.nlp(text)
        indices = []
        subjects = []
        for token in doc:
            # noun of a sentence or noun in a phrase
            if (token.dep_ in ["nsubj", "nsubjpass"]) or (token.dep_ == "ROOT" and token.pos_ in ["NOUN", "PROPN"]):
                indices.append(token.i)
                subjects.append(token.text)
        if never_empty and len(indices) == 0:
            indices.append(0)
            subjects.append(doc[0].text)
        return indices, subjects


def torch_fix_seed(seed=42):
    # Python random
    random.seed(seed)
    # Numpy
    np.random.seed(seed)
    # Pytorch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms = True
