import pandas as pd
import numpy as np

from spacy.tokenizer import Tokenizer

import readability
import spacy

import warnings
warnings.filterwarnings('ignore')

def mean(list):
    return sum(list) / len(list)


def extract_lenght_of_parag(dataframe, split='\n'):
    """
    df = (dataframe) Dataframe to extract lenght of paragraph
    split = (string type. reversed slash + n or '. ') reversed slash + n for long paragraph, '. ' for short paragraph
    """
    dataframe = dataframe
    split = split
    ax = []
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)
    for i in dataframe['excerpt']:  # Extract lenght of long paragraph
        tmp = i.split(split)
        tmp_lenght = []
        for x, sent in enumerate(tokenizer.pipe(tmp)):
            sent_token = [token.text for token in sent]
            tmp_lenght += [len(sent_token)]
        ax += [round(mean(tmp_lenght))]
    return ax


def cal_read_o_time(words):
    words = words
    tmp = len(words) / 200
    return round(tmp, 2)


def cal_total_read_o_time(dataframe, split='\n'):
    """
    df = (dataframe) Dataframe to extract lenght of Read-O-Time
    split = (string type. reversed slash + n or '. ') reversed slash + n for long paragraph, '. ' for short paragraph
    """
    ax = []
    nlp = spacy.load("en_core_web_sm")
    tokenizer = Tokenizer(nlp.vocab)
    for i in dataframe['excerpt']:  # Extract lenght of long paragraph
        tmp = i.split(split)
        tmp_lenght = []
        for x, sent in enumerate(tokenizer.pipe(tmp)):
            sent_token = [token.text for token in sent]
            tmp_lenght += [cal_read_o_time(sent_token)]
        ax += [round(mean(tmp_lenght), 2)]
    return ax


# source: https://www.kaggle.com/ravishah1/readability-feature-engineering-non-nn-baseline/data
def readability_measurements(passage: str):
    """
    This function uses the readability library for feature engineering.
    It includes textual statistics, readability scales and metric, and some pos stats
    """
    passage = passage
    results = readability.getmeasures(passage, lang='en')

    chars_per_word = results['sentence info']['characters_per_word']
    syll_per_word = results['sentence info']['syll_per_word']
    words_per_sent = results['sentence info']['words_per_sentence']

    kincaid = results['readability grades']['Kincaid']
    ari = results['readability grades']['ARI']
    coleman_liau = results['readability grades']['Coleman-Liau']
    flesch = results['readability grades']['FleschReadingEase']
    gunning_fog = results['readability grades']['GunningFogIndex']
    lix = results['readability grades']['LIX']
    smog = results['readability grades']['SMOGIndex']
    rix = results['readability grades']['RIX']
    dale_chall = results['readability grades']['DaleChallIndex']

    tobeverb = results['word usage']['tobeverb']
    auxverb = results['word usage']['auxverb']
    conjunction = results['word usage']['conjunction']
    pronoun = results['word usage']['pronoun']
    preposition = results['word usage']['preposition']
    nominalization = results['word usage']['nominalization']

    pronoun_b = results['sentence beginnings']['pronoun']
    interrogative = results['sentence beginnings']['interrogative']
    article = results['sentence beginnings']['article']
    subordination = results['sentence beginnings']['subordination']
    conjunction_b = results['sentence beginnings']['conjunction']
    preposition_b = results['sentence beginnings']['preposition']

    return [chars_per_word, syll_per_word, words_per_sent,
            kincaid, ari, coleman_liau, flesch, gunning_fog, lix, smog, rix, dale_chall,
            tobeverb, auxverb, conjunction, pronoun, preposition, nominalization,
            pronoun_b, interrogative, article, subordination, conjunction_b, preposition_b]


def spacy_features(df: pd.DataFrame):
    """
    This function generates features using spacy en_core_wb_lg
    I learned about this from these resources:
    https://www.kaggle.com/konradb/linear-baseline-with-cv
    https://www.kaggle.com/anaverageengineer/comlrp-baseline-for-complete-beginners
    """
    df = df

    nlp = spacy.load('en_core_web_lg')
    with nlp.disable_pipes():
        vectors = np.array([nlp(text).vector for text in df.excerpt])

    return vectors


def get_spacy_col_names():
    names = list()
    for i in range(300):
        names.append(f"spacy_{i}")

    return names

def generate_other_features(passage: str):
    """
    This function is where I test miscellaneous features
    This is experimental
    """
    # punctuation count
    periods = passage.count(".")
    commas = passage.count(",")
    semis = passage.count(";")
    exclaims = passage.count("!")
    questions = passage.count("?")

    # Some other stats
    num_char = len(passage)
    num_words = len(passage.split(" "))
    unique_words = len(set(passage.split(" ")))
    word_diversity = unique_words / num_words

    word_len = [len(w) for w in passage.split(" ")]
    longest_word = np.max(word_len)
    avg_len_word = np.mean(word_len)

    return [periods, commas, semis, exclaims, questions,
            num_char, num_words, unique_words, word_diversity,
            longest_word, avg_len_word]

class CLRDataset:
    """
    This is my CommonLit Readability Dataset.
    By calling the get_df method on an object of this class,
    you will have a fully feature engineered dataframe
    """

    def __init__(self, df: pd.DataFrame, train: bool, n_folds=2):
        self.df = df
        self.excerpts = df["excerpt"]

        self._extract_features()

    def _extract_features(self):
        scores_df = pd.DataFrame(self.df["excerpt"].apply(lambda p: readability_measurements(p)).tolist(),
                                 columns=["chars_per_word", "syll_per_word", "words_per_sent",
                                          "kincaid", "ari", "coleman_liau", "flesch", "gunning_fog", "lix", "smog",
                                          "rix", "dale_chall",
                                          "tobeverb", "auxverb", "conjunction", "pronoun", "preposition",
                                          "nominalization",
                                          "pronoun_b", "interrogative", "article", "subordination", "conjunction_b",
                                          "preposition_b"])
        self.df = pd.merge(self.df, scores_df, left_index=True, right_index=True)

        other_df = pd.DataFrame(self.df["excerpt"].apply(lambda p: generate_other_features(p)).tolist(),
                                columns=["periods", "commas", "semis", "exclaims", "questions",
                                         "num_char", "num_words", "unique_words", "word_diversity",
                                         "longest_word", "avg_len_word"])
        self.df = pd.merge(self.df, other_df, left_index=True, right_index=True)

    def get_df(self):
        return self.df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int):
        pass

def predict_text(text):
    text = pd.DataFrame(data=[text], columns=['excerpt'])
    text = CLRDataset(text, False)
    dataframe = text.get_df()
    dataframe['paragraph_avg_rot'] = cal_total_read_o_time(dataframe, '\n')
    dataframe['sentence_avg_rot'] = cal_total_read_o_time(dataframe, '. ')
    columns_except = ['excerpt']

    dataframe = dataframe.drop(columns=columns_except)
    pred_y = dataframe['ari']
    words_len = dataframe['num_words']
    rot = dataframe['paragraph_avg_rot']
    conj = dataframe['conjunction']
    voca_div = dataframe['word_diversity']
    longest_word = dataframe['longest_word']
    unique_word = dataframe['unique_words']
    return pred_y, words_len, rot, conj, voca_div, longest_word, unique_word