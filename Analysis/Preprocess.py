import docx2txt
import pandas as pd
import os


def process_year(year_string):
    if '‡' in year_string:
        year1 = year_string.split('‡')[0][1:]
        year2 = year_string.split('‡')[1][:-1]
        return [year1, year2]
    else:
        return [year_string[1:-1]]  # remove brackets


def preprocess_song(song_corpus):
    # Translate
    pass


def split_text_into_songs(full_corpus, start_word, end_word):
    corpus = str(start_word + full_corpus.split(start_word)[1].split(end_word)[0])
    tomes = corpus.split("\nSTARTTOM ")
    tome_texts = []
    for tome in tomes[1:]:
        tome_split = [str.strip() for str in tome.splitlines() if str != '']
        tome_next = "\n".join(tome_split[1:])
        tome_num = int(tome_split[0])
        tome_year = process_year(tome_split[2].split()[0])  # returns list
        tome_year_from = tome_year_to = tome_year[0]
        if len(tome_year) == 2:
            tome_year_to = tome_year[1]
        tome_name_split = tome_next.split("IMETOM")
        tome_name = ' '.join(tome_name_split[0].split('\n')).strip()
        tome_text_f = [line for line in tome_name_split[1].split('\n') if line != '']
        pod_tom_split = tome_name_split[1].split("PODTOM")
        if len(pod_tom_split) == 1:
            tome_texts.append({'tome_name': tome_name, 'pod_tome_name': tome_name, 'text': tome_text_f,
                               'tome_year_from': tome_year_from, 'tome_year_to': tome_year_to,
                               'tome_num': tome_num, 'pod_tome_num': 1})
        else:
            counter = 1
            for i in range(1, len(pod_tom_split), 2):
                pod_tom_ime = pod_tom_split[i].strip()
                pod_tom_tekst = [str.strip() for str in pod_tom_split[i + 1].splitlines() if str != '']
                tome_texts.append(
                    {'tome_name': tome_name, 'pod_tome_name': pod_tom_ime, 'text': pod_tom_tekst,
                     'tome_year_from': tome_year_from, 'tome_year_to': tome_year_to,
                     'tome_num': tome_num, 'pod_tome_num': counter})
                counter += 1

    for tome_text in tome_texts:
        text_in = tome_text['text']
        found_start = False
        start = 0
        end = 0
        pesni = []

        # Split songs
        for i in range(len(text_in)):
            line = text_in[i]
            if str.isupper(line) or (str.isnumeric(line.split('.')[0]) and line.endswith(".")):
                if not found_start:
                    start = i
                    found_start = True
                else:
                    end = i
                    if end - start > 1:
                        pesni.append(text_in[start:end])
                    start = end
        pesni.append(text_in[start:])
        tome_text['pesni'] = pesni

    return tome_texts


if __name__ == "__main__":
    dataset_path = "Data"
    tome_1 = os.path.join(dataset_path, "Koneski_1.docx")
    tome_2 = os.path.join(dataset_path, "Koneski_2.docx")

    text_tome_1 = docx2txt.process(tome_1)
    dictionary_tome_1 = split_text_into_songs(text_tome_1, start_word="POCETOKOT", end_word="KRAJOT")

    text_tome_2 = docx2txt.process(tome_2)
    dictionary_tome_2 = split_text_into_songs(text_tome_2, start_word="POCETOKOT", end_word="KRAJOT")

    # {'tome_name': tome_name, 'pod_tome_name': pod_tom_ime, 'text': pod_tom_tekst,
    # 'tome_year': tome_year, 'tome_num': tome_num, 'pod_tome_num': counter})
    final_dict = {'Godina_od': [], 'Godina_do': [], 'Zbirka': [], 'Zbirka_broj': [], 'Podzbirka': [],
                  'Podzbirka_broj': [], 'Pesna_Broj': [], 'Pesna_Ime': [], 'Pesna': [], 'Stih_Broj': [], 'Stih': []}

    for item in dictionary_tome_1:
        pesni = item['pesni']
        counter_pesna = 1
        for pesna in pesni:
            counter_stih = 1
            ime_pesna = ""
            for stih in pesna:
                if counter_stih == 1:
                    ime_pesna = stih
                final_dict['Godina_od'].append(item['tome_year_from'])
                final_dict['Godina_do'].append(item['tome_year_to'])
                final_dict['Zbirka'].append(item['tome_name'])
                final_dict['Zbirka_broj'].append(item['tome_num'])
                final_dict['Podzbirka'].append(item['pod_tome_name'])
                final_dict['Podzbirka_broj'].append(item['pod_tome_num'])
                final_dict['Pesna_Broj'].append(str(counter_pesna))
                final_dict['Pesna_Ime'].append(ime_pesna)
                final_dict['Pesna'].append('\n'.join(pesna))
                final_dict['Stih_Broj'].append(str(counter_stih))
                final_dict['Stih'].append(stih)
                counter_stih += 1
            counter_pesna += 1

    for item in dictionary_tome_2:
        pesni = item['pesni']
        counter_pesna = 1
        for pesna in pesni:
            counter_stih = 1
            ime_pesna = ""
            for stih in pesna:
                if counter_stih == 1:
                    ime_pesna = stih
                final_dict['Godina_od'].append(item['tome_year_from'])
                final_dict['Godina_do'].append(item['tome_year_to'])
                final_dict['Zbirka'].append(item['tome_name'])
                final_dict['Zbirka_broj'].append(item['tome_num'])
                final_dict['Podzbirka'].append(item['pod_tome_name'])
                final_dict['Podzbirka_broj'].append(item['pod_tome_num'])
                final_dict['Pesna_Broj'].append(str(counter_pesna))
                final_dict['Pesna_Ime'].append(ime_pesna)
                final_dict['Pesna'].append('\n'.join(pesna))
                final_dict['Stih_Broj'].append(str(counter_stih))
                final_dict['Stih'].append(stih)
                counter_stih += 1
            counter_pesna += 1

    final_dataset = pd.DataFrame.from_dict(final_dict)
    final_dataset.to_csv("Data/Dataset.csv", encoding='utf-8')

    # spacy.load('mk_core_news_lg')
