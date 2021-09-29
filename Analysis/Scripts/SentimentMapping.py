import os
import pickle
import pandas as pd
import nltk
from nltk.corpus import stopwords
import re
import spacy

if __name__ == "__main__":
    stop_words = set(stopwords.words('english'))
    nlp = spacy.load('mk_core_news_lg')
    lemmatizer = nltk.WordNetLemmatizer()
    with open('../stop_words_mk.txt', encoding='utf-8') as stop_file:
        stop_words_lines = stop_file.readlines()
    stop_words_mk = list(set([line.rstrip() for line in stop_words_lines]))
    def_dic = {
        'neutral': -1,
        'positive': -1,
        'negative': -1,
        'anger': -1,
        'anticipation': -1,
        'disgust': -1,
        'fear': -1,
        'joy': -1,
        'sadness': -1,
        'surprise': -1,
        'trust': -1
    }
    dataset_path = "../Data"

    # dict_path = "../SentimentData/dictionary.pkl"

    # WITH LEMMATIZED DICT
    dict_path = "../SentimentData/dictionary_lemma.pkl"

    dataset_path = os.path.join(dataset_path, "Koneski_final_finished_dataset.csv")
    file = open(dict_path, "rb")
    sentiment_mapping = pickle.load(file)
    df = pd.read_csv(dataset_path, usecols=range(0, 14), index_col=0)

    # Statistics now
    final_df_list_eng = []
    final_df_list_mk = []

    songs_df = df.groupby(by=["Godina_od", "Zbirka_broj", "Pesna_Broj"], as_index=False).first()[
        ["Godina_od", "Godina_do", "Pesna", "Pesna_eng", "Zbirka", "Podzbirka", "Pesna_Ime"]]

    for _, row in songs_df.iterrows():
        vector_V_eng = []
        vector_V_mk = []
        vector_A_eng = []
        vector_A_mk = []
        vector_D_eng = []
        vector_D_mk = []
        vector_emotions_eng = []
        vector_emotions_mk = []
        vector_emotional_attendence_eng = []
        vector_emotional_attendence_mk = []

        pesna_ime = row["Pesna_Ime"]
        zbirka_ime = row["Zbirka"]
        podzbirka_ime = row["Podzbirka"]
        godina_od = row["Godina_od"]
        godina_do = row["Godina_do"]
        pesna_mk = row["Pesna"]
        pesna_eng = row["Pesna_eng"]

        print(f'Currently processing song: {pesna_ime} from tome {zbirka_ime} from subtome {podzbirka_ime} composed '
              f'in the time period ({godina_od},{godina_do}).')

        # Cleaning english
        zborovi_eng = pesna_eng.split()
        zborovi_eng = [re.sub(r'[^\w\s]', '', word) for word in zborovi_eng]
        zborovi_eng = [word for word in zborovi_eng if word not in stop_words]
        zborovi_eng = [word for word in zborovi_eng if word.isalpha()]
        zborovi_eng = [lemmatizer.lemmatize(each_word).lower() for each_word in zborovi_eng]

        # Cleaning macedonian
        zborovi_mk = pesna_mk.split()
        zborovi_mk = [re.sub(r'[^\w\s]', '', word) for word in zborovi_mk]
        zborovi_mk = [word for word in zborovi_mk if word not in stop_words_mk]
        zborovi_mk = [word.lower() for word in zborovi_mk if word.isalpha()]
        # processed_song_mk = zborovi_mk
        processed_song_mk = []
        for word in zborovi_mk:
            doc = nlp(word)
            for each_word in doc:
                lemma = each_word.lemma_
            processed_song_mk.append(lemma)

        vad_len_eng = vad_len_mk = 0
        sentiment_len_eng = sentiment_len_mk = 0
        for token in zborovi_eng:
            if token in sentiment_mapping['eng']:
                V = sentiment_mapping['eng'][token]['valence']
                A = sentiment_mapping['eng'][token]['arousal']
                D = sentiment_mapping['eng'][token]['dominance']
                emotional_intensity_vector = sentiment_mapping['eng'][token]['emotional_intensity']
                emotional_attendence_vector = [1 if emotional_intensity_vector[val] > 0 else 0
                                               for val in emotional_intensity_vector]
                if sum(emotional_attendence_vector) > 0:
                    for k, v in emotional_intensity_vector.items():
                        if v == -1:
                            emotional_intensity_vector[k] = 0

                vector_V_eng.append(V)
                vector_A_eng.append(A)
                vector_D_eng.append(D)
                if sum(emotional_attendence_vector) != 0:
                    vector_emotions_eng.append(emotional_intensity_vector)
                    vector_emotional_attendence_eng.append(emotional_attendence_vector)
                    sentiment_len_eng += 1
                vad_len_eng += 1

        for token in processed_song_mk:
            if token in sentiment_mapping['mk']:
                V = sentiment_mapping['mk'][token]['valence']
                A = sentiment_mapping['mk'][token]['arousal']
                D = sentiment_mapping['mk'][token]['dominance']
                emotional_intensity_vector = sentiment_mapping['mk'][token]['emotional_intensity']
                emotional_attendence_vector = [1 if emotional_intensity_vector[val] > 0 else 0
                                               for val in emotional_intensity_vector]
                if sum(emotional_attendence_vector) > 0:
                    for k, v in emotional_intensity_vector.items():
                        if v == -1:
                            emotional_intensity_vector[k] = 0

                vector_V_mk.append(V)
                vector_A_mk.append(A)
                vector_D_mk.append(D)
                if sum(emotional_attendence_vector) != 0:
                    vector_emotions_mk.append(emotional_intensity_vector)
                    vector_emotional_attendence_mk.append(emotional_attendence_vector)
                    sentiment_len_mk += 1
                vad_len_mk += 1

        lens_eng = [sum(i) for i in zip(*vector_emotional_attendence_eng)]
        lens_mk = [sum(i) for i in zip(*vector_emotional_attendence_mk)]

        if len(lens_eng) == 0:
            lens_eng = [0] * 11
        if len(lens_mk) == 0:
            lens_mk = [0] * 11

        V_mean_eng = sum([v for v in vector_V_eng if v != -1]) / vad_len_eng if vad_len_eng != 0 else 0
        V_mean_mk = sum([v for v in vector_V_mk if v != -1]) / vad_len_mk if vad_len_mk != 0 else 0

        A_mean_eng = sum([v for v in vector_A_eng if v != -1]) / vad_len_eng if vad_len_eng != 0 else 0
        A_mean_mk = sum([v for v in vector_A_mk if v != -1]) / vad_len_mk if vad_len_mk != 0 else 0

        D_mean_eng = sum([v for v in vector_D_eng if v != -1]) / vad_len_eng if vad_len_eng != 0 else 0
        D_mean_mk = sum([v for v in vector_D_mk if v != -1]) / vad_len_mk if vad_len_mk != 0 else 0

        anger_mean_eng = sum([v['anger'] for v in vector_emotions_eng]) / lens_eng[3] if lens_eng[3] != 0 else 0
        anger_mean_mk = sum([v['anger'] for v in vector_emotions_mk]) / lens_mk[3] if lens_mk[3] != 0 else 0

        anticipation_mean_eng = sum(
            [v['anticipation'] for v in vector_emotions_eng]) / lens_eng[4] if lens_eng[4] != 0 else 0
        anticipation_mean_mk = sum(
            [v['anticipation'] for v in vector_emotions_mk]) / lens_mk[4] if lens_mk[4] != 0 else 0

        disgust_mean_eng = sum([v['disgust'] for v in vector_emotions_eng]) / lens_eng[5] if lens_eng[5] != 0 else 0
        disgust_mean_mk = sum([v['disgust'] for v in vector_emotions_mk]) / lens_mk[5] if lens_mk[5] != 0 else 0

        fear_mean_eng = sum([v['fear'] for v in vector_emotions_eng]) / lens_eng[6] if lens_eng[6] != 0 else 0
        fear_mean_mk = sum([v['fear'] for v in vector_emotions_mk]) / lens_mk[6] if lens_mk[6] != 0 else 0

        joy_mean_eng = sum([v['joy'] for v in vector_emotions_eng]) / lens_eng[7] if lens_eng[7] != 0 else 0
        joy_mean_mk = sum([v['joy'] for v in vector_emotions_mk]) / lens_mk[7] if lens_mk[7] != 0 else 0

        sadness_mean_eng = sum([v['sadness'] for v in vector_emotions_eng]) / lens_eng[8] if lens_eng[8] != 0 else 0
        sadness_mean_mk = sum([v['sadness'] for v in vector_emotions_mk]) / lens_mk[8] if lens_mk[8] != 0 else 0

        surprise_mean_eng = sum([v['surprise'] for v in vector_emotions_eng]) / lens_eng[9] if lens_eng[9] != 0 else 0
        surprise_mean_mk = sum([v['surprise'] for v in vector_emotions_mk]) / lens_mk[9] if lens_mk[9] != 0 else 0

        trust_mean_eng = sum([v['trust'] for v in vector_emotions_eng]) / lens_eng[10] if lens_eng[10] != 0 else 0
        trust_mean_mk = sum([v['trust'] for v in vector_emotions_mk]) / lens_mk[10] if lens_mk[10] != 0 else 0

        final_dict_eng = {
            'Godina_od': godina_od,
            'Godina_do': godina_do,
            'Zbirka': zbirka_ime,
            'Podzbirka': podzbirka_ime,
            'Pesna_Ime': pesna_ime,
            'Pesna_eng': pesna_eng,
            'V_mean': V_mean_eng,
            'A_mean': A_mean_eng,
            'D_mean': D_mean_eng,
            'Neutral_word_ratio': float(lens_eng[0] / vad_len_eng),
            'Positive_word_ratio': float(lens_eng[1] / vad_len_eng),
            'Negative_word_ratio': float(lens_eng[2] / vad_len_eng),
            'Anger_word_ratio': float(lens_eng[3] / vad_len_eng),
            'Anger_mean_intensity': anger_mean_eng,
            'Anticipation_word_ratio': float(lens_eng[4] / vad_len_eng),
            'Anticipation_mean_intensity': anticipation_mean_eng,
            'Disgust_word_ration': float(lens_eng[5] / vad_len_eng),
            'Disgust_mean_intensity': disgust_mean_eng,
            'Fear_word_ratio': float(lens_eng[6] / vad_len_eng),
            'Fear_mean_intensity': fear_mean_eng,
            'Joy_word_ratio': float(lens_eng[7] / vad_len_eng),
            'Joy_mean_intensity': joy_mean_eng,
            'Sadness_word_ratio': float(lens_eng[8] / vad_len_eng),
            'Sadness_mean_intensity': sadness_mean_eng,
            'Surprise_word_ratio': float(lens_eng[9] / vad_len_eng),
            'Surprise_mean_intensity': surprise_mean_eng,
            'Trust_word_ratio': float(lens_eng[10] / vad_len_eng),
            'Trust_mean_intensity': trust_mean_eng
        }

        final_dict_mk = {
            'Godina_od': godina_od,
            'Godina_do': godina_do,
            'Zbirka': zbirka_ime,
            'Podzbirka': podzbirka_ime,
            'Pesna_Ime': pesna_ime,
            'Pesna_mk': pesna_mk,
            'V_mean': V_mean_mk,
            'A_mean': A_mean_mk,
            'D_mean': D_mean_mk,
            'Neutral_word_ratio': float(lens_mk[0] / vad_len_mk),
            'Positive_word_ratio': float(lens_mk[1] / vad_len_mk),
            'Negative_word_ratio': float(lens_mk[2] / vad_len_mk),
            'Anger_word_ratio': float(lens_mk[3] / vad_len_mk),
            'Anger_mean_intensity': anger_mean_mk,
            'Anticipation_word_ratio': float(lens_mk[4] / vad_len_mk),
            'Anticipation_mean_intensity': anticipation_mean_mk,
            'Disgust_word_ration': float(lens_mk[5] / vad_len_mk),
            'Disgust_mean_intensity': disgust_mean_mk,
            'Fear_word_ratio': float(lens_mk[6] / vad_len_mk),
            'Fear_mean_intensity': fear_mean_mk,
            'Joy_word_ratio': float(lens_mk[7] / vad_len_mk),
            'Joy_mean_intensity': joy_mean_mk,
            'Sadness_word_ratio': float(lens_mk[8] / vad_len_mk),
            'Sadness_mean_intensity': sadness_mean_mk,
            'Surprise_word_ratio': float(lens_mk[9] / vad_len_mk),
            'Surprise_mean_intensity': surprise_mean_mk,
            'Trust_word_ratio': float(lens_mk[10] / vad_len_mk),
            'Trust_mean_intensity': trust_mean_mk
        }

        final_df_list_eng.append(final_dict_eng)
        final_df_list_mk.append(final_dict_mk)

    df_eng = pd.DataFrame(final_df_list_eng)
    # df_eng.to_csv("../SentimentResultSets/SentimentEng.csv", encoding='utf-8', index=False)
    # With lemma dict
    df_eng.to_csv("../SentimentResultSets/SentimentEngLemma.csv", encoding='utf-8', index=False)

    df_mk = pd.DataFrame(final_df_list_mk)
    # df_mk.to_csv("../SentimentResultSets/SentimentMk.csv", encoding='utf-8', index=False)
    # With lemma dict
    df_mk.to_csv("../SentimentResultSets/SentimentMkLemma.csv", encoding='utf-8', index=False)
