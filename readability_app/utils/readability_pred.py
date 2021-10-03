from readability import Readability
import pandas as pd
import numpy as np

import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('pos_tag')
nltk.download('word_tags_sent')

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import pos_tag

from spacy.tokenizer import Tokenizer

import readability
import spacy

from sklearn import model_selection

import warnings
warnings.filterwarnings('ignore')
import re

import xgboost

def preprocessing(dataframe):
    tmp = dataframe.apply(lambda e: e.replace('\n', ''))
    excerpt_processed = []
    for e in tmp:
        # find alphabets
        e = re.sub("[^a-zA-Z]", " ", e)

        # convert to lower case
        e = e.lower()

        # tokenize words
        e = nltk.word_tokenize(e)

        # remove stopwords
        e = [word for word in e if not word in set(stopwords.words("english"))]

        # lemmatization
        lemma = nltk.WordNetLemmatizer()
        e = [lemma.lemmatize(word) for word in e]
        e = " ".join(e)

        excerpt_processed.append(e)

    dataframe['processed_exerpt'] = excerpt_processed
    return dataframe


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


def pos_tag_features(passage: str):
    """
    This function counts the number of times different parts of speech occur in an excerpt
    """
    pos_tags = ["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
                "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO", "UH",
                "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"]

    tags = pos_tag(word_tokenize(passage))
    tag_list = list()

    for tag in pos_tags:
        tag_list.append(len([i[0] for i in tags if i[1] == tag]))

    return tag_list


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


def create_folds(data: pd.DataFrame, num_splits: int):
    """
    This function creates a kfold cross validation system based on this reference:
    https://www.kaggle.com/abhishek/step-1-create-folds
    """
    # we create a new column called kfold and fill it with -1
    data["kfold"] = -1

    # the next step is to randomize the rows of the data
    data = data.sample(frac=1, random_state=42).reset_index(drop=True)

    # calculate number of bins by Sturge's rule
    # I take the floor of the value, you can also
    # just round it
    num_bins = int(np.floor(1 + np.log2(len(data))))

    # bin targets
    data.loc[:, "bins"] = pd.cut(
        data["target"], bins=num_bins, labels=False
    )

    # initiate the kfold class from model_selection module
    kf = model_selection.StratifiedKFold(n_splits=num_splits)

    # fill the new kfold column
    # note that, instead of targets, we use bins!
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, 'kfold'] = f

    # drop the bins column
    data = data.drop("bins", axis=1)

    # return dataframe with folds
    return data


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

        if train:
            self.df = create_folds(self.df, n_folds)

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

        spacy_df = pd.DataFrame(spacy_features(self.df), columns=get_spacy_col_names())
        self.df = pd.merge(self.df, spacy_df, left_index=True, right_index=True)

        pos_df = pd.DataFrame(self.df["excerpt"].apply(lambda p: pos_tag_features(p)).tolist(),
                              columns=["CC", "CD", "DT", "EX", "FW", "IN", "JJ", "JJR", "JJS", "LS", "MD",
                                       "NN", "NNS", "NNP", "NNPS", "PDT", "POS", "PRP", "RB", "RBR", "RBS", "RP", "TO",
                                       "UH",
                                       "VB", "VBD", "VBG", "VBZ", "WDT", "WP", "WRB"])
        self.df = pd.merge(self.df, pos_df, left_index=True, right_index=True)

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
    text_len = text.split()
    assert len(text_len) >= 100
    try:
        text_tmp = preprocessing(text)
        text = CLRDataset(text_tmp, False)
        text.dataframe = text.get_df()
        text.dataframe['paragraph_avg_rot'] = cal_total_read_o_time(text.dataframe, '\n')
        text.dataframe['sentence_avg_rot'] = cal_total_read_o_time(text.dataframe, '. ')
        text.dataframe['total_avg_rot'] = [cal_read_o_time(i) for i in text.dataframe['excerpt']]
        columns_except = ['id', 'excerpt', 'processed_exerpt']

        text.column = text.column.drop(columms=columns_except)
        model = xgboost.XGBRegressor(n_jobs=-1, booster='gbtree', random_state=42, n_estimators=100, verbosity=0)
        pred_y = model.predict(text.dataframe)

        return pred_y
    except:
        return "최소 100글자의 글을 입력해야 합니다."