from nltk import pos_tag
from difflib import SequenceMatcher
from collections import Counter
import numpy as np
import spacy
from nltk.corpus import wordnet
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from rapidfuzz import fuzz

nlp = spacy.load("en_core_web_sm")

# Features to Extract From Question Pairs
def len_ratio(tokens1, tokens2):
    len1 = len(tokens1)
    len2 = len(tokens2)
    return min(len1, len2) / max(len1, len2) if max(len1, len2) > 0 else 0

def lcs_length(seq1, seq2):
    m, n = len(seq1), len(seq2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if seq1[i - 1] == seq2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

def compute_pos_features(tokens1, tokens2):
    pos_tags1 = [tag for _, tag in pos_tag(tokens1)]
    pos_tags2 = [tag for _, tag in pos_tag(tokens2)]
    # Compute LCS
    pos_lcs = lcs_length(pos_tags1, pos_tags2) / max(len(pos_tags1), len(pos_tags2)) if max(len(pos_tags1), len(pos_tags2)) > 0 else 0
    
    # Compute Jaccard
    pos_set1, pos_set2 = set(pos_tags1), set(pos_tags2)
    jaccard = len(pos_set1 & pos_set2) / len(pos_set1 | pos_set2) if len(pos_set1 | pos_set2) > 0 else 0

    # Compute Cosine
    freq1, freq2 = Counter(pos_tags1), Counter(pos_tags2)
    all_tags = set(freq1.keys()).union(set(freq2.keys()))
    vec1, vec2 = np.array([freq1[tag] for tag in all_tags]), np.array([freq2[tag] for tag in all_tags])
    cosine = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)) if np.linalg.norm(vec1) and np.linalg.norm(vec2) else 0

    return pos_lcs, jaccard, cosine

def dep_sim(tokens1, tokens2):
    doc1 = nlp(" ".join(tokens1))
    doc2 = nlp(" ".join(tokens2))
    deps1 = Counter([token.dep_ for token in doc1])
    deps2 = Counter([token.dep_ for token in doc2])
    all_deps = set(deps1.keys()).union(set(deps2.keys()))
    vec1 = np.array([deps1.get(dep, 0) for dep in all_deps])
    vec2 = np.array([deps2.get(dep, 0) for dep in all_deps])
    if np.linalg.norm(vec1) == 0 or np.linalg.norm(vec2) == 0:
        return 0
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


def char_ngram(tokens1, tokens2, n=3):
    def ngrams(text, n):
        return set([text[i:i+n] for i in range(len(text) - n + 1)])
    text1 = " ".join(tokens1)
    text2 = " ".join(tokens2)
    ngrams1 = ngrams(text1, n)
    ngrams2 = ngrams(text2, n)
    intersection = ngrams1.intersection(ngrams2)
    union = ngrams1.union(ngrams2)
    return len(intersection) / len(union) if union else 0


def uniq_word_cnt(tokens1, tokens2):
    words1 = set(tokens1)
    words2 = set(tokens2)
    unique_count = len(words1.symmetric_difference(words2))
    union_count = len(words1.union(words2))
    if union_count == 0:
        return 1
    else:
        return 1 - (unique_count / union_count)
    

def precompute_synonyms(vocabulary):
    return {word: get_synonyms(word) for word in vocabulary}

def get_synonyms(word):
    return {lemma.name() for syn in wordnet.synsets(word) for lemma in syn.lemmas()}

def synonym_similarity(tokens1, tokens2, synonym_dict):
    overlap = sum(1 for token in tokens1 if synonym_dict.get(token, set()).intersection(tokens2))
    denom = max(len(tokens1), len(tokens2))
    return overlap / denom if denom > 0 else 0

def sequence_alignment(tokens1, tokens2):
    seq_match = SequenceMatcher(None, tokens1, tokens2)
    return seq_match.ratio()

def sentiment_difference_vader(text1, text2):
    analyzer = SentimentIntensityAnalyzer()
    sentiment1 = analyzer.polarity_scores(text1)['compound'] 
    sentiment2 = analyzer.polarity_scores(text2)['compound']  
    difference = abs(sentiment1 - sentiment2)
    return 1 - (difference / 2)

def fuzzy_similarity_score(text1, text2):
    return fuzz.ratio(text1, text2)

def extract_features(row, synonym_dict):
    tokens1 = row['question1_tokens']
    tokens2 = row['question2_tokens']
    text1 = row['question1']
    text2 = row['question2']
    
    lcs_length, jaccard, cosine = compute_pos_features(tokens1, tokens2)
    output_vector = [
        len_ratio(tokens1, tokens2),
        lcs_length,
        dep_sim(tokens1, tokens2),
        char_ngram(tokens1, tokens2),
        uniq_word_cnt(tokens1, tokens2),
        cosine,
        jaccard,
        synonym_similarity(tokens1, tokens2, synonym_dict),
        sequence_alignment(tokens1, tokens2),
        sentiment_difference_vader(text1, text2),
        fuzzy_similarity_score(text1, text2)
    ] 
    return output_vector

feature_names = ["len_ratio", "pos_lcs", "dep_sim", "char_ngram", "uniq_cmp", "pos_cos", "pos_jac", "syn_sim", "seq_align", "sent_diff", "fuzz_sim"]
