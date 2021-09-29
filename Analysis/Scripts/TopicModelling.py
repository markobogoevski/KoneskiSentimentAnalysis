import nltk
import spacy
import pandas as pd
import gensim
import os
import re
import pickle
import pyLDAvis.gensim
from nltk.corpus import stopwords
from spacy.lang.en import English
from gensim import corpora

with open('../stop_words_mk.txt', encoding='utf-8') as stop_file:
    stop_words_lines = stop_file.readlines()
stop_words_mk = list(set([line.rstrip() for line in stop_words_lines]))

parser = English()
stop_words = set(stopwords.words('english'))
nlp = spacy.load('mk_core_news_lg')
lemmatizer = nltk.WordNetLemmatizer()


def tokenize(text, lang='en'):
    lda_tokens = []
    if lang == 'en':
        tokens = parser(text)
        for token in tokens:
            if token.orth_.isspace():
                continue
            elif token.like_url:
                lda_tokens.append('URL')
            elif token.orth_.startswith('@'):
                lda_tokens.append('SCREEN_NAME')
            else:
                lda_tokens.append(token.lower_)
    else:
        tokens = nltk.word_tokenize(text)
        for token in tokens:
            if token.isspace():
                continue
            else:
                lda_tokens.append(token.lower())
    return lda_tokens


def prepare_song(song_text, lang='en'):
    tokens = tokenize(song_text, lang)
    tokens = [token for token in tokens if len(token) > 2]
    tokens = [re.sub(r'[^\w\s]', '', token) for token in tokens]
    tokens = [token for token in tokens if token.isalpha()]
    if lang == 'mk':
        tokens = [token for token in tokens if token not in stop_words_mk]
        processed_tokens = []
        for token in tokens:
            doc = nlp(token)
            for each_word in doc:
                lemma = each_word.lemma_
            processed_tokens.append(lemma)
        return processed_tokens
    else:
        tokens = [token for token in tokens if token not in stop_words]
        tokens = [lemmatizer.lemmatize(token) for token in tokens]
        return tokens


def define_lda_and_topics(corpus, dictionary, num_topics, num_words_per_topic, lang='en'):
    ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=num_topics, id2word=dictionary, passes=15,
                                               minimum_probability=0.001)
    model_name = f'{lang}_gensim_topic_model_{str(num_topics)}_topics.gensim'
    model_path = os.path.join('../GensimTopicModels', model_name)
    ldamodel.save(model_path)
    topics = ldamodel.print_topics(num_words=num_words_per_topic)
    return ldamodel, topics


if __name__ == "__main__":
    dataset_path = "../Data"
    dataset_path = os.path.join(dataset_path, "Koneski_final_finished_dataset.csv")
    df = pd.read_csv(dataset_path, usecols=range(0, 14), index_col=0)
    songs_df = df.groupby(by=["Godina_od", "Zbirka_broj", "Pesna_Broj"], as_index=False).first()[
        ["Godina_od", "Godina_do", "Pesna", "Pesna_eng", "Zbirka", "Podzbirka", "Pesna_Ime"]]

    songs_tokenized_eng = []
    songs_tokenized_mk = []

    for _, row in songs_df.iterrows():
        pesna_ime = row["Pesna_Ime"]
        zbirka_ime = row["Zbirka"]
        podzbirka_ime = row["Podzbirka"]
        godina_od = row["Godina_od"]
        godina_do = row["Godina_do"]
        pesna_mk = row["Pesna"]
        pesna_eng = row["Pesna_eng"]
        tokens_eng = prepare_song(pesna_eng)
        tokens_mk = prepare_song(pesna_mk, lang='mk')
        songs_tokenized_eng.append(tokens_eng)
        songs_tokenized_mk.append(tokens_mk)

    dictionary_eng = corpora.Dictionary(songs_tokenized_eng)
    dictionary_mk = corpora.Dictionary(songs_tokenized_mk)
    corpus_eng = [dictionary_eng.doc2bow(text) for text in songs_tokenized_eng]
    corpus_mk = [dictionary_mk.doc2bow(text) for text in songs_tokenized_mk]
    pickle.dump(corpus_eng, open('../TopicData/topic_corpus_eng.pkl', 'wb'))
    pickle.dump(corpus_mk, open('../TopicData/topic_corpus_mk.pkl', 'wb'))
    dictionary_eng.save('../TopicData/topic_dictionary_eng.gensim')
    dictionary_mk.save('../TopicData/topic_dictionary_mk.gensim')
    num_topics = 15
    num_words_per_topic = 10

    # ENG LDA
    lda_eng, topics_eng = define_lda_and_topics(corpus=corpus_eng, dictionary=dictionary_eng, num_topics=num_topics,
                                                num_words_per_topic=num_words_per_topic, lang='en')

    # MK LDA
    lda_mk, topics_mk = define_lda_and_topics(corpus=corpus_mk, dictionary=dictionary_mk, num_topics=num_topics,
                                              num_words_per_topic=num_words_per_topic, lang='mk')

    # LOADING
    # lda_eng = gensim.models.ldamodel.LdaModel.load('GensimTopicModels/en_gensim_topic_model_15_topics.gensim')
    # lda_mk = gensim.models.ldamodel.LdaModel.load('GensimTopicModels/mk_gensim_topic_model_15_topics.gensim')
    # topics_eng = lda_eng.print_topics(num_words=num_words_per_topic)
    # topics_mk = lda_mk.print_topics(num_words=num_words_per_topic)
    # dictionary_eng = gensim.corpora.dictionary.Dictionary.load('TopicData/topic_dictionary_eng.gensim')
    # dictionary_mk = gensim.corpora.dictionary.Dictionary.load('TopicData/topic_dictionary_mk.gensim')
    # eng_pick = open('TopicData/topic_corpus_eng.pkl', "rb")
    # corpus_eng = pickle.load(eng_pick)
    # eng_pick.close()
    # mk_pick = open('TopicData/topic_corpus_mk.pkl', "rb")
    # corpus_mk = pickle.load(mk_pick)
    # mk_pick.close()

    lines_to_write = []
    lines_to_write.append(f"Generated {num_topics} topics for english language using {str(num_words_per_topic)} "
                          f"number of words per topic: \n\n")
    for topic_no, topic in enumerate(topics_eng):
        lines_to_write.append(f'Topic {str(topic_no + 1)}: \n {topic}\n')
    lines_to_write.append('\n\n')
    lines_to_write.append(f"Generated {num_topics} topics for macedonian language using {str(num_words_per_topic)} "
                          f"number of words per topic: \n\n")
    for topic_no, topic in enumerate(topics_mk):
        lines_to_write.append(f'Topic {str(topic_no + 1)}: \n {topic}')
    lines_to_write.append('\n\n')
    lines_to_write.append("============================================================================================"
                          "==========================\n")
    lines_to_write.append("============================================================================================"
                          "==========================\n")
    lines_to_write.append("============================================================================================"
                          "==========================\n")
    lines_to_write.append("============================================================================================"
                          "==========================\n\n")
    lines_to_write.append("SONGS: \n")

    for _, row in songs_df.iterrows():
        pesna_ime = row["Pesna_Ime"]
        zbirka_ime = row["Zbirka"]
        podzbirka_ime = row["Podzbirka"]
        godina_od = row["Godina_od"]
        godina_do = row["Godina_do"]
        pesna_mk = row["Pesna"]
        pesna_eng = row["Pesna_eng"]
        tokens_eng = prepare_song(pesna_eng)
        tokens_mk = prepare_song(pesna_mk, lang='mk')
        tokens_eng_bow = dictionary_eng.doc2bow(tokens_eng)
        tokens_mk_bow = dictionary_mk.doc2bow(tokens_mk)
        topics_eng_doc = lda_eng.get_document_topics(tokens_eng_bow)
        best_3_topics_eng = sorted(topics_eng_doc, key=lambda x: -x[1])[:3]
        topics_mk_doc = lda_mk.get_document_topics(tokens_mk_bow)
        best_3_topics_mk = sorted(topics_mk_doc, key=lambda x: -x[1])[:3]

        lines_to_write.append(
            f'ZBIRKA: {zbirka_ime}, PODZBIRKA: {podzbirka_ime}, IME NA PESNA: {pesna_ime}, napisana vo periodot'
            f'od {str(godina_od)} do {str(godina_do)}.\n')
        lines_to_write.append("==============================\n")
        lines_to_write.append("NA ANGLISKI: \n")
        lines_to_write.append(pesna_eng)
        lines_to_write.append("\n")
        lines_to_write.append(f"BEST TOPICS: {best_3_topics_eng}\n")
        lines_to_write.append(f"DESCRIPTION OF BEST TOPICS: \n")
        for topic in best_3_topics_eng:
            lines_to_write.append("===============\n")
            lines_to_write.append(f'Topic number: {topic[0]}, topic probability: {topic[1]}\n')
            lines_to_write.append(f'Topic description: {topics_eng[topic[0]]}\n')
        lines_to_write.append("==============================\n")
        lines_to_write.append("NA MAKEDONSKI: \n")
        lines_to_write.append(pesna_mk)
        lines_to_write.append("\n")
        lines_to_write.append(f"NAJSOODVETNI TOPICS: {best_3_topics_mk}\n")
        lines_to_write.append(f"OPIS NA NAJDOBRITE TOPICS: \n")
        for topic in best_3_topics_mk:
            lines_to_write.append("===============\n")
            lines_to_write.append(f'Broj na tema: {topic[0]}, verojatnost za temata: {topic[1]}\n')
            lines_to_write.append(f'Topic description: {topics_mk[topic[0]]}\n')
        lines_to_write.append("==============================\n\n")

    lines_to_write.append("===========================================================\n")
    lines_to_write.append("===========================================================\n")
    lines_to_write.append("===========================================================\n")
    lines_to_write.append("===========================================================\n")
    with open('../TopicResults/topic_modelling.txt', 'w', encoding='utf-8') as f:
        f.writelines(lines_to_write)

    # Visualization
    lda_display_eng = pyLDAvis.gensim.prepare(lda_eng, corpus_eng, dictionary_eng, sort_topics=False)
    pyLDAvis.save_html(lda_display_eng, f'../TopicResults/lda_eng_{num_topics}_{num_words_per_topic}.html')

    lda_display_mk = pyLDAvis.gensim.prepare(lda_mk, corpus_mk, dictionary_mk, sort_topics=False)
    pyLDAvis.save_html(lda_display_mk, f'../TopicResults/lda_mk_{num_topics}_{num_words_per_topic}.html')
