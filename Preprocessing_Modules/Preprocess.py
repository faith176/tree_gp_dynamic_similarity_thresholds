

import os
import scipy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
from .Features import extract_features
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sentence_transformers import SentenceTransformer
import spacy
import numpy as np
import swifter
from Preprocessing_Modules.Features import precompute_synonyms


stop_words = set(stopwords.words('english'))
bert_model = SentenceTransformer('all-MiniLM-L6-v2')
lemmatizer = WordNetLemmatizer()
nlp = spacy.load("en_core_web_sm")


def preprocess_text(text, remove_stopwords=True, apply_lemmatization=True):
    if isinstance(text, str):
        tokens = word_tokenize(text.lower())
        tokens = [word for word in tokens if word.isalnum()]
        if remove_stopwords:
            tokens = [word for word in tokens if word not in stop_words]
        if apply_lemmatization:
            tokens = [lemmatizer.lemmatize(word) for word in tokens]
        return tokens
    return []


def get_similarity(text1, text2):
    return max(0, 1 - scipy.spatial.distance.cosine(bert_model.encode(text1) , bert_model.encode(text2)))


def create_dataset(subset_percent = 0.01, num_rows = None):
    tqdm.pandas()
    filename = f"Data/Preprocessed_Files/processed_{subset_percent}.pkl"
    if os.path.isfile(filename):
        df = pd.read_pickle(filename)
        print("File found and loaded into DataFrame:")
    else:
        # Handle the case where the file doesn't exist
        print(f"File '{filename}' does not exist.")
        file_path = os.path.join('data', 'quora_duplicate_questions.csv')
        df = pd.read_csv(file_path)
        num_rows_to_keep = int(len(df) * subset_percent)
        df = df.iloc[:num_rows_to_keep]
        df.dropna(subset=['question1', 'question2'], inplace=True)
        df = df[["is_duplicate", "question1", "question2"]]

        print("Preprocessing Text")
        df['question1_tokens'] = df['question1'].swifter.apply(preprocess_text)
        df['question2_tokens'] = df['question2'].swifter.apply(preprocess_text)
        
        print("Creating Synonym Dict")
        all_tokens = set(token for tokens in df['question1_tokens'].tolist() + df['question2_tokens'].tolist() for token in tokens)
        synonym_dict = precompute_synonyms(all_tokens)
        
        print("Extracting Features")
        df['evolve_features'] = df.swifter.apply(lambda row: extract_features(row, synonym_dict), axis=1)

        print("Calculating similarity")
        df['similarity'] = df.swifter.apply(lambda row: get_similarity(row['question1'], row['question2']), axis=1)
        
        print("Creating pickle file")
        df.to_pickle(filename)
    if num_rows is not None:
        df = df.sample(n=num_rows, random_state=42).reset_index(drop=True)
    return df

def extract_X_Y_SIM(df):
    X = df['evolve_features']
    Y = df['is_duplicate']
    SIMILARITY = df['similarity']

    X = np.array(X.tolist())
    Y = Y.values
    SIMILARITY = SIMILARITY.values
    return X, Y, SIMILARITY

def extract_test_train(df, test_size=0.2):
    X, Y, SIMILARITY = extract_X_Y_SIM(df)

    # Train-test split
    num_samples = len(X)
    indices = list(range(num_samples))
    train_indices, test_indices = train_test_split(
        indices, test_size=test_size, random_state=42, stratify=Y
    )

    # Separate training and test sets
    X_train = X[train_indices]
    X_test = X[test_indices]
    Y_train = Y[train_indices]
    Y_test = Y[test_indices]
    SIMILARITY_train = SIMILARITY[train_indices]
    SIMILARITY_test = SIMILARITY[test_indices]

    # Min-Max Scaling: Fit on training data, apply to both
    scaler = MinMaxScaler()
    X_train = scaler.fit_transform(X_train) 
    X_test = scaler.transform(X_test)    
    
    print(f"Training Size: {len(X_train)}")  
    print(f"Test Size: {len(X_test)}")  
    print("\n")

    return X, Y, test_indices, [X_train, X_test, Y_train, Y_test, SIMILARITY_train, SIMILARITY_test]

