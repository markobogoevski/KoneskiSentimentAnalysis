import os
import pandas as pd
import copy
import spacy
import nltk
from nltk.corpus import stopwords
import pickle
import re

# Napraven so materijali od https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm?fbclid=IwAR23987lAMj6L8SBp0Sm1442hrIKen5KEHtYwh8SvHdeyz7V5LOayKINFRg

if __name__ == "__main__":
    stop_words_eng = set(stopwords.words('english'))
    with open('../stop_words_mk.txt', encoding='utf-8') as stop_file:
        stop_words_lines = stop_file.readlines()
    stop_words_mk = list(set([line.rstrip() for line in stop_words_lines]))
    nlp = spacy.load('mk_core_news_lg')
    lemmatizer = nltk.WordNetLemmatizer()
    path = "../SentimentData"
    emotion_intensity_path = os.path.join(path, "EmotionIntensity.txt")
    vad_path = os.path.join(path, "VAD.txt")
    emotion_map_path = os.path.join(path, "EmotionMapCSV.csv")

    word_emotion_dictionary = {'eng': dict(), 'mk': dict()}
    # Dictionary in the format
    # {'eng': {word:
    #   'emotional_intensity': {emotion_1:intensity_1,...,emotion_8:intensity_8},
    #   'valence':valence_number, (-1 if no value)
    #   'arousal':arousal_number,
    #   'dominance':dominance_number},
    #   'mk' : ...}

    with open(emotion_intensity_path, 'r', encoding='utf-8') as intensity_file:
        with open(vad_path, 'r', encoding='utf-8') as vad_file:
            # READING INTENSITY FILE
            lines_int = intensity_file.readlines()
            intensity_lines = [line.split('\t') for line in lines_int]
            intensity_labels = intensity_lines[0]
            set_words_int_eng = set([line[0].lower() for line in intensity_lines[1:]])
            set_words_int_mk = set([line[1].lower() for line in intensity_lines[1:] if line[1] != 'NO TRANSLATION'])

            # READING VAD FILE
            lines_vad = vad_file.readlines()
            vad_lines = [line.split('\t') for line in lines_vad]
            vad_labels = vad_lines[0]
            set_words_vad_eng = set([line[0].lower() for line in vad_lines[1:]])
            set_words_vad_mk = set([line[1].lower() for line in vad_lines[1:] if line[1] != 'NO TRANSLATION'])

            # READING EMOTIONAL MAP FILE
            emotion_map_df = pd.read_csv(emotion_map_path, encoding='utf-8')
            wanted_columns = ['English (en)', 'Macedonian (mk)', 'Positive', 'Negative', 'Anger', 'Anticipation',
                              'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']
            emotion_df = emotion_map_df[wanted_columns]
            emotion_df = emotion_df[emotion_df['English (en)'].notnull()]
            eng_words = emotion_df.iloc[:, 0].tolist()
            mk_words = emotion_df.iloc[:, 1].tolist()
            set_words_emotion_eng = set([word.lower() for word in eng_words])
            set_words_emotion_mk = set([word.lower() for word in mk_words if word != 'NO TRANSLATION'])

            print(f'Number of english words with emotional intensity: ', len(set_words_int_eng))
            print(f'Number of macedonian words with emotional intensity: ', len(set_words_int_mk))
            print(f'Number of english words with VAD values: ', len(set_words_vad_eng))
            print(f'Number of macedonian words with VAD values: ', len(set_words_vad_mk))
            print(f'Number of english words with emotional maps: ', len(set_words_emotion_eng))
            print(f'Number of macedonian words with emotional maps: ', len(set_words_emotion_mk))
            print(f'Number of english words with VAD values and emotional intensities and emotional maps: ',
                  len((set_words_emotion_eng.intersection(set_words_int_eng)).intersection(set_words_vad_eng)))
            print(f'Number of macedonian words with VAD values and emotional intensities and emotional maps: ',
                  len((set_words_emotion_mk.intersection(set_words_int_mk)).intersection(set_words_vad_mk)))

            print("Creating dictionary...")
            default_word_dict = {
                'emotional_intensity': {
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
                },
                'valence': -1,
                'arousal': -1,
                'dominance': -1}

            for line in intensity_lines[1:]:
                eng_stop = False
                mk_stop = False
                eng_word = line[0].lower()
                mk_word = line[1].lower()
                if eng_word in stop_words_eng:
                    eng_stop = True
                if mk_word in stop_words_mk:
                    mk_stop = True

                # WITH LEMMATIZATION
                # Clean eng word
                eng_word = re.sub(r'[^\w\s]', '', eng_word)
                eng_word = lemmatizer.lemmatize(eng_word)

                # Clean mk word
                mk_word = re.sub(r'[^\w\s]', '', mk_word)
                doc = nlp(mk_word)
                for word in doc:
                    mk_word = word.lemma_

                emotion = line[2].lower()
                intensity = float(line[3].rstrip())

                if not eng_stop:
                    if eng_word not in word_emotion_dictionary['eng']:
                        word_emotion_dictionary['eng'][eng_word] = copy.deepcopy(default_word_dict)
                    word_emotion_dictionary['eng'][eng_word]['emotional_intensity'][emotion] = intensity

                if not mk_stop:
                    if mk_word != 'no translation':
                        if mk_word not in word_emotion_dictionary['mk']:
                            word_emotion_dictionary['mk'][mk_word] = copy.deepcopy(default_word_dict)
                        word_emotion_dictionary['mk'][mk_word]['emotional_intensity'][emotion] = intensity

            for line in vad_lines[1:]:
                eng_stop = False
                mk_stop = False
                eng_word = line[0].lower()
                mk_word = line[1].lower()
                if eng_word in stop_words_eng:
                    eng_stop = True
                if mk_word in stop_words_mk:
                    mk_stop = True

                # WITH LEMMATIZATION
                # Clean eng word
                eng_word = re.sub(r'[^\w\s]', '', eng_word)
                eng_word = lemmatizer.lemmatize(eng_word)
                # Clean mk word
                mk_word = re.sub(r'[^\w\s]', '', mk_word)
                doc = nlp(mk_word)
                for word in doc:
                    mk_word = word.lemma_

                V = float(line[2])
                A = float(line[3])
                D = float(line[4])

                if not eng_stop:
                    if eng_word not in word_emotion_dictionary['eng']:
                        word_emotion_dictionary['eng'][eng_word] = copy.deepcopy(default_word_dict)
                    word_emotion_dictionary['eng'][eng_word]['valence'] = V
                    word_emotion_dictionary['eng'][eng_word]['arousal'] = A
                    word_emotion_dictionary['eng'][eng_word]['dominance'] = D

                if not mk_stop:
                    if mk_word != 'no translation':
                        if mk_word not in word_emotion_dictionary['mk']:
                            word_emotion_dictionary['mk'][mk_word] = copy.deepcopy(default_word_dict)
                        word_emotion_dictionary['mk'][mk_word]['valence'] = V
                        word_emotion_dictionary['mk'][mk_word]['arousal'] = A
                        word_emotion_dictionary['mk'][mk_word]['dominance'] = D

            emotions = [column.lower() for column in emotion_df.columns[2:]]
            for _, row in emotion_df.iterrows():
                eng_stop = False
                mk_stop = False
                eng_word = row['English (en)'].lower()
                mk_word = row['Macedonian (mk)'].lower()
                if eng_word in stop_words_eng:
                    eng_stop = True
                if mk_word in stop_words_mk:
                    mk_stop = True

                # WITH LEMMATIZATION
                # Clean eng word
                eng_word = re.sub(r'[^\w\s]', '', eng_word)
                eng_word = lemmatizer.lemmatize(eng_word)
                # Clean mk word
                mk_word = re.sub(r'[^\w\s]', '', mk_word)
                doc = nlp(mk_word)
                for word in doc:
                    mk_word = word.lemma_

                emotion_vector = row[2:].tolist()

                if eng_word not in word_emotion_dictionary['eng'] and not eng_stop:
                    word_emotion_dictionary['eng'][eng_word] = copy.deepcopy(default_word_dict)

                if not mk_stop:
                    if mk_word != 'no translation':
                        if mk_word not in word_emotion_dictionary['mk']:
                            word_emotion_dictionary['mk'][mk_word] = copy.deepcopy(default_word_dict)

                idx = 0
                neutral = 1
                for emotion in emotions:
                    if not eng_stop and word_emotion_dictionary['eng'][eng_word]['emotional_intensity'][emotion] == -1:
                        word_emotion_dictionary['eng'][eng_word]['emotional_intensity'][emotion] = emotion_vector[idx]
                    if not mk_stop:
                        if mk_word != 'no translation':
                            if word_emotion_dictionary['mk'][mk_word]['emotional_intensity'][emotion] == -1:
                                word_emotion_dictionary['mk'][mk_word]['emotional_intensity'][emotion] = emotion_vector[
                                    idx]
                    if not eng_stop and word_emotion_dictionary['eng'][eng_word]['emotional_intensity'][emotion] > 0:
                        neutral = 0
                    idx += 1

                if not eng_stop:
                    word_emotion_dictionary['eng'][eng_word]['emotional_intensity']['neutral'] = neutral
                if not mk_stop:
                    if mk_word != 'no translation':
                        word_emotion_dictionary['mk'][mk_word]['emotional_intensity']['neutral'] = neutral

            # Saving the dictionary
            # dictionary_file = open("../SentimentData/dictionary.pkl", "wb")
            dictionary_file = open("../SentimentData/dictionary_lemma.pkl", "wb")
            pickle.dump(word_emotion_dictionary, dictionary_file)
            dictionary_file.close()
        vad_file.close()
    intensity_file.close()
