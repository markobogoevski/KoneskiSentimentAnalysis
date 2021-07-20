import os
import pandas as pd

if __name__ == "__main__":
    path = "Data"
    dataset_path = os.path.join(path, "Koneski_final - Dataset.csv")
    dataset = pd.read_csv(dataset_path, index_col=0)

    index = 0
    while index < len(dataset):
        pesna = str(dataset.iloc[index]['Pesna_eng']).rstrip()
        stihovi = pesna.split('\n')
        for stih in stihovi:
            if stih != '\n' or stih != '':
                dataset['Stih_eng'][index] = stih
                index += 1

    new_path = os.path.join(path, "Koneski_final_finished_dataset.csv")
    dataset.to_csv(new_path, encoding='utf-8')
