import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import RSLPStemmer
from nltk.classify import apply_features
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score
)

from string import punctuation

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('rslp')

class Processor:
    def __init__(self):
        self.comments = []
        self.stopwords = set(stopwords.words('portuguese') + list(punctuation))
        self.stemmer = RSLPStemmer()
        self.all_words = []
        self.frequency = []
        self.examples = []

    def train(self):
        self.read()
        self.process_content()
        self.review_content()

        base_classification = apply_features(self.review_words, self.comments)
        classificator = nltk.NaiveBayesClassifier.train(base_classification)

        return classificator

    def read(self):
        self.comments = pd.read_csv('comments.csv')

    def tokenize(self, text):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens]

        return tokens

    def clean_stopword(self, tokens):
        filtered_tokens = [word for word in tokens if
                            word not in self.stopwords]

        return filtered_tokens

    def stemming(self, tokens):
        tokens_stem = [self.stemmer.stem(word) for word in tokens]

        return tokens_stem

    def process_content(self):
        self.comments['comment'] = self.comments['comment'].apply(self.tokenize)
        self.comments['comment'] = self.comments['comment'].apply(self.clean_stopword)
        self.comments['comment'] = self.comments['comment'].apply(self.stemming)

        aux_comments = self.comments
        self.comments = []
        for _, row in aux_comments.iterrows():
            self.comments.append((row['comment'], row['emotion']))


    def review_content(self):
        for (tokens, emotion) in self.comments:
            self.all_words.extend(tokens)

        self.frequency = nltk.FreqDist(self.all_words)

    def review_words(self, words):
        words = set(words)
        stats = {}

        for common_words in self.frequency:
            stats['%s' % common_words] = (common_words in words)

        return stats

    def convert_word(self, word):
        example = self.review_words(
            self.stemming(
                self.clean_stopword(
                    self.tokenize(word)
                )
            )
        )

        return example

    def stats(self, classificator):
        if not self.examples:
            return {}

        y_true = [label for (features, label) in self.examples]
        y_pred = [classificator.classify(features) for (features, label) in
                  self.examples]
        labels = sorted(set(y_true))

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, labels=labels,
                                         average='macro', zero_division=0),
            "recall": recall_score(y_true, y_pred, labels=labels,
                                   average='macro', zero_division=0),
            "f1_score": f1_score(y_true, y_pred, labels=labels, average='macro',
                                 zero_division=0),
            "confusion_matrix": confusion_matrix(y_true, y_pred,
                                                 labels=labels).tolist(),
            "classification_report": classification_report(
                y_true, y_pred, labels=labels, output_dict=True, zero_division=0
            )
        }

        return metrics