import numpy as np
from itertools import product
import Diffraction_generation_script as dgs
import pandas as pd
import pickle
import matplotlib.pyplot as plt

cristals = ["\"Ag.laz\"", "\"Al.laz\"", "\"Al2O3_sapphire.laz\"", "\"Au.laz\"", "\"B4C.laz\"", "\"Ba.laz\"", "\"Be.laz\"", "\"BeO.laz\"", "\"C_diamond.laz\"", "\"C_graphite.laz\"", "\"Cr.laz\"", "\"Cs.laz\"", "\"Cu.laz\"", "\"Cu2MnAl.laz\"", "\"Fe.laz\"", "\"Ga.laz\"", "\"Gd.laz\"", "\"Ge.laz\"", "\"H2O_ice_1h.laz\"", "\"He4_hcp.laz\"", "\"Hg.laz\"", "\"I2.laz\"", "\"K.laz\"", "\"Li.laz\"", "\"LiF.laz\"", "\"Mo.laz\"", "\"Na2Ca3Al2F14.laz\"", "\"Nb.laz\"", "\"Ni.laz\"", "\"Pb.laz\"", "\"Pt.laz\"", "\"Rb.laz\"", "\"Si.laz\"", "\"Ti.laz\"", "\"Tl.laz\"", "\"UO2.laz\"", "\"Zn.laz\"", "\"Y2O3.laz\""]

stats = [1E7, 2E7, 5E7, 1E8, 2E8, 5E8]

pulce_duration = np.linspace(10, 300, 20)

arrays = [cristals, stats, pulce_duration]

ans = list(product(*arrays))

print(len(ans))

# Короче ans это масиив из списков параметров на создание датасета ('"Ag.laz"', 10000000.0, 10.0), ('"Ag.laz"', 10000000.0, 25.263157894736842)....
# Его нужно будет распарсить на 38 ядер как раз этого достаточно будет. Там по 1200 сканов на 1 ядро получится
# Далее идет код который уже генерирует то что нужно)


mcrun_path = "/Applications/McStas-3.3.app/Contents/Resources/mcstas/3.3/bin/"
mcstas_path = "/Applications/McStas-3.3.app/Contents/Resources/mcstas/3.3/"

do_dif = dgs.RenderPredicts(mcrun_path, mcstas_path)

# Создаем пустой DataFrame
df = pd.DataFrame(columns=['Matrix', 'Crystal', 'Stats', 'Pulce duration'])

for i in range(len(cristals)):

    Diffraction_matrix = do_dif.get_diffraction(cristals[i], 2E7, 300)

    # Создание строки данных
    new_row = {'Matrix': Diffraction_matrix, 'Crystal': cristals[i], 'Stats': 2E7, 'Pulce duration': 300}

    # Добавление новой строки в DataFrame
    df = df.append(new_row, ignore_index=True)

# Сохранение DataFrame в файл с использованием pickle
# Тут как обычно сделаешь много датасетов, я их потом объединю
with open('dataset_sequoia.pkl', 'wb') as f:
    pickle.dump(df, f)

"""
# Загрузка DataFrame из файла
with open('dataset.pkl', 'rb') as f:
    df_loaded = pickle.load(f)

print(df_loaded.head())

# Получение первой матрицы из загруженного DataFrame
first_matrix = df_loaded['Matrix'][0]

# Построение тепловой карты
plt.figure(figsize=(10, 8))
plt.imshow(first_matrix, cmap='viridis', aspect='auto', norm='log')
plt.colorbar()
plt.title('Heatmap of the First Matrix')
plt.show()
"""