import torch
import random
import numpy as np
from nltk.translate.bleu_score import sentence_bleu

def find_divisors(n):
    """
    Finds all the divisors of a number n.
    """
    divisors = [i for i in range(1, n+1) if n % i == 0]
    return divisors

def set_seed(seed):
    """
    Sets the seed for all random number generators.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bleu_score(pred, tgt):
    """
    Calculates the BLEU score for the predicted caption compared to the target caption.
    """
    bleu1 = sentence_bleu([tgt], pred, weights=(1, 0, 0, 0))
    bleu2 = sentence_bleu([tgt], pred, weights=(0.5, 0.5, 0, 0))
    bleu3 = sentence_bleu([tgt], pred, weights=(0.33, 0.33, 0.33, 0))
    bleu4 = sentence_bleu([tgt], pred, weights=(0.25, 0.25, 0.25, 0.25))
    return bleu1, bleu2, bleu3, bleu4