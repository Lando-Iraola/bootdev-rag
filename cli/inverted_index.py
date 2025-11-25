import json
import math
import os
import pickle
import string
from collections import Counter

from nltk.stem import PorterStemmer
from search_utils import BM25_B, BM25_K1, load_stopwords


class InvertedIndex:
    def __init__(self):
        self.index = {}
        self.docmap = {}
        self.term_frequencies = {}
        self.doc_lengths = {}

    def __tokenize(self, text):
        punctuation_map = {}
        for p in string.punctuation:
            punctuation_map[p] = None
        no_punctuation_table = str.maketrans(punctuation_map)

        stop_words = load_stopwords()
        tokens = text.lower().translate(no_punctuation_table).split()
        tokens = [item for item in tokens if item not in stop_words]
        stemmer = PorterStemmer()
        new_tokens = []
        for token in tokens:
            if not token:
                continue
            new_tokens.append(stemmer.stem(token))
        return new_tokens

    def __add_document(self, doc_id, text):
        tokens = self.__tokenize(text)
        self.doc_lengths[doc_id] = len(tokens)
        self.term_frequencies[doc_id] = Counter()
        for token in tokens:
            self.term_frequencies[doc_id][token] += 1
            if token not in self.index:
                self.index[token] = set(
                    [
                        doc_id,
                    ]
                )
                continue

            self.index[token].add(doc_id)

    def get_document(self, term):
        token = self.__tokenize(term)[0]
        if token not in self.index:
            return []
        return sorted(list(self.index[token]))

    def build(self):
        with open(
            "./data/movies.json",
            "r",
        ) as movies:
            data_set = json.load(movies)
            for movie in data_set["movies"]:
                text = f"{movie['title']} {movie['description']}"
                self.docmap[movie["id"]] = movie
                self.__add_document(movie["id"], text)

    def save(self):
        cache_dir = "cache"
        os.makedirs(cache_dir, exist_ok=True)
        index_file = os.path.join(cache_dir, "index.pkl")
        with open(index_file, "wb") as i_f:
            pickle.dump(self.index, i_f)
        map_file = os.path.join(cache_dir, "docmap.pkl")
        with open(map_file, "wb") as m_f:
            pickle.dump(self.docmap, m_f)
        freq_file = os.path.join(cache_dir, "term_frequencies.pkl")
        with open(freq_file, "wb") as f_f:
            pickle.dump(self.term_frequencies, f_f)

        doc_length_file = os.path.join(cache_dir, "doc_lengths.pkl")
        with open(doc_length_file, "wb") as d_l_f:
            pickle.dump(self.doc_lengths, d_l_f)

    def load(self):
        cache_dir = "cache"
        index_file = os.path.join(cache_dir, "index.pkl")
        map_file = os.path.join(cache_dir, "docmap.pkl")
        freq_file = os.path.join(cache_dir, "term_frequencies.pkl")
        doc_length_file = os.path.join(cache_dir, "doc_lengths.pkl")

        if not os.path.exists(index_file):
            raise Exception("No index file to load")

        if not os.path.exists(map_file):
            raise Exception("No map file to load")

        if not os.path.exists(freq_file):
            raise Exception("No term frequency file to load")

        with open(index_file, "rb") as i_f:
            self.index = pickle.load(i_f)

        with open(map_file, "rb") as m_f:
            self.docmap = pickle.load(m_f)

        with open(freq_file, "rb") as f_f:
            self.term_frequencies = pickle.load(f_f)

        with open(doc_length_file, "rb") as d_l_f:
            self.doc_lengths = pickle.load(d_l_f)

    def get_tf(self, doc_id, term):
        token = self.__tokenize(term)
        if len(token) > 1:
            raise Exception("More than one term was given for frequency search")
        if token[0] not in self.term_frequencies[doc_id]:
            return 0
        return self.term_frequencies[doc_id][token[0]]

    def get_bm25_idf(self, term: str):
        if len(term.split()) > 1:
            raise Exception("More than one term was given for IDF")
        n = len(self.docmap)
        df = len(self.get_document(term))

        return math.log((n - df + 0.5) / (df + 0.5) + 1)

    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        length_norm = (
            1 - b + b * (self.doc_lengths[doc_id] / self.__get_avg_doc_length())
        )

        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    def __get_avg_doc_length(self):
        lengths = 0
        count = 0
        for id in self.doc_lengths:
            lengths += self.doc_lengths[id]
            count += 1
        if count == 0:
            return 0.0
        print("avg:", lengths / count)
        return lengths / count
