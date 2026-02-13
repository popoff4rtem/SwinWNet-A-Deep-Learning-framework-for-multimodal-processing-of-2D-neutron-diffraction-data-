import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle

"""
# Загрузка DataFrame из файла
with open('/Users/artempopov/PycharmProjects/ML_Project/DifClassification/dataset.pkl', 'rb') as f:
    df = pickle.load(f)

"""

class Qwrapper:
    def __init__(self, thettta_resolution, Lambda_resolution):
        self.thettta_resolution = thettta_resolution
        self.Lambda_resolution = Lambda_resolution

    def getQDIFADC(self, matrix, num_channels):

        rad_coeff = 0.0174533
        thetta = np.linspace(-170,170, self.thettta_resolution)
        L = np.linspace(0.1, 10, self.Lambda_resolution)
        Q = []
        I = []

        for rows in range(len(matrix)):
            for cols in range(len(matrix[rows])):
                #q = L[rows] * np.sin(abs(rad_coeff * thetta[cols]))/ (4*np.pi)
                #q = 4*np.pi / (np.sin(abs(rad_coeff * thetta[cols])) * L[rows])
                #q = L[rows] / (np.sin(abs(rad_coeff * thetta[cols])) / 4*np.pi)
                #q = L[rows] / (np.sin(abs(rad_coeff * thetta[cols]/2)) / 2)
                q = L[rows] / (np.sin(abs(rad_coeff * thetta[cols]/2)) * 2)
                Q.append(q)
                I.append(abs(matrix[rows][cols]))

        # Определение диапазона и количества каналов
        qmin = np.min(Q)
        qmax = np.max(Q)
        #num_channels = 2024

        # Определение границ каналов
        channels = np.linspace(qmin, qmax, num_channels + 1)

        # Определение индексов каналов для каждого значения Q
        channel_indices = np.digitize(Q, channels) - 1

        # Инициализация массива для хранения сумм I по каналам
        I_summed = np.zeros(num_channels)

        # Накопление значений I в соответствующих каналах
        for i in range(len(Q)):
            if 0 <= channel_indices[i] < num_channels:
                I_summed[channel_indices[i]] += I[i]

        # Построение графика
        channel_centers = (channels[:-1] + channels[1:]) / 2  # центры каналов

        return channel_centers, I_summed
    
    def getQDIFSORT(self, matrix):

        rad_coeff = 0.0174533
        thetta = np.linspace(-170,170, self.thettta_resolution)
        L = np.linspace(0.1, 10, self.Lambda_resolution)
        Q = []
        I = []

        for rows in range(len(matrix)):
            for cols in range(len(matrix[rows])):
                #q = L[rows] * np.sin(abs(rad_coeff * thetta[cols]))/ (4*np.pi)
                q = 4*np.pi / (np.sin(abs(rad_coeff * thetta[cols])) * L[rows])
                Q.append(q)
                I.append(abs(matrix[rows][cols]))

        graph = zip(Q,I)

        #отсортируем, взяв первый элемент каждого списка как ключ
        graph_s = sorted(graph, key=lambda tup: tup[0])

        Qs = [x[0] for x in graph_s]
        Is = [x[1] for x in graph_s]

        return Qs, Is
    

class Dwrapper:
    def __init__(self):
        pass

    def transform_Q_to_D(q, I):
        d = 1/q

        return d, I
    
"""
plotter = Qwrapper()

Q, I = plotter.getQDIFADC(df['Matrix'][23], 512)

plt.figure(figsize=(10, 6))
plt.plot(Q, I, label='I(Q)')
plt.xlabel('Q')
plt.ylabel('Summed I')
plt.title('I(Q) with 4096 Channels')
#plt.yscale('log')
plt.legend()
plt.show()
"""
"""
plt.figure(figsize=(10, 6))
plt.plot(channel_centers, I_summed, label='I(Q)')
plt.plot(Qs, Is, label='Is(Qs)')
plt.xlabel('Q')
plt.ylabel('Summed I')
plt.title('I(Q) with 4096 Channels')
#plt.yscale('log')
plt.legend()
plt.show()
"""