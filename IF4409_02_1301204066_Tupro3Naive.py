#=====================================================================
#                              LIBRARY
#=====================================================================
import pandas as pd
import math
import time
from statistics import mean
#=====================================================================
#                           MEMBACA DATA 
#=====================================================================
data = pd.read_excel("traintest.xlsx", sheet_name='train')
dataTest = pd.read_excel("traintest.xlsx", sheet_name='test').drop('y', axis=1)
#=====================================================================

def decide(data):
    '''
    Fungsi ini digunakan untuk menetapkan label y = 1 dan y = 0
    '''
    y0 = []
    y1 = []
    for i in range(len(data)):
        if data['y'][i] == 0: y0.append([data['x1'][i], data['x2'][i], data['x3'][i], data['y'][i]])
        elif data['y'][i] == 1: y1.append([data['x1'][i], data['x2'][i], data['x3'][i], data['y'][i]])
    return y0, y1

def Mean_Variansi(data):
    '''
    Fungsi ini digunakan untuk menghitung mean dan variansi dari tiap label
    '''
    y0, y1 = decide(data)

    data0 = pd.DataFrame(data = y0, columns = ['x1', 'x2', 'x3', 'y'])
    data1 = pd.DataFrame(data = y1, columns = ['x1', 'x2', 'x3', 'y'])

    Mean0 = data0[['x1', 'x2', 'x3']].mean()
    Mean1 = data1[['x1', 'x2', 'x3']].mean()

    Variansi0 = data0[['x1', 'x2', 'x3']].var()
    Variansi1 = data1[['x1', 'x2', 'x3']].var()
    
    return data0, data1, Mean0, Mean1, Variansi0, Variansi1

def Bayes(data, dataTest):
    '''
    Fungsi ini digunakan untuk menghitung bayesian menggunakan gaussian model
    '''
    hasil = []
    data0, data1, Mean0, Mean1, Variansi0, Variansi1 = Mean_Variansi(data)
    for i in range(len(dataTest)):
        n = ((1/(Variansi0['x1'] * math.sqrt(2 * math.pi))) * math.exp(-(dataTest['x1'][i] - Mean0['x1'])/(2 * Variansi0['x1']**2)) * 
             (1/(Variansi0['x2'] * math.sqrt(2 * math.pi))) * math.exp(-(dataTest['x2'][i] - Mean0['x2'])/(2 * Variansi0['x2']**2)) * 
             (1/(Variansi0['x3'] * math.sqrt(2 * math.pi))) * math.exp(-(dataTest['x3'][i] - Mean0['x3'])/(2 * Variansi0['x3']**2)) * (len(data0)/len(data)))
        y = ((1/(Variansi1['x1'] * math.sqrt(2 * math.pi))) * math.exp(-(dataTest['x1'][i] - Mean1['x1'])/(2 * Variansi1['x1']**2)) * 
             (1/(Variansi1['x2'] * math.sqrt(2 * math.pi))) * math.exp(-(dataTest['x2'][i] - Mean1['x2'])/(2 * Variansi1['x2']**2)) *  
             (1/(Variansi1['x3'] * math.sqrt(2 * math.pi))) * math.exp(-(dataTest['x3'][i] - Mean1['x3'])/(2 * Variansi1['x3']**2)) * (len(data1)/len(data)))
             
        # Menyimpan model training         
        if y > n: hasil.append([dataTest.loc[i, 'id'], 1])
        elif n > y: hasil.append([dataTest.loc[i, 'id'], 0])
    hasil = pd.DataFrame(data = hasil, columns = ['id', 'y'])
    return hasil

def Evaluasi(data):
    '''
    Fungsi ini digunakan untuk melakukan validasi terhadap data menggunakan 5-Fold Cross Validation 
    '''
    datasetfold1 = data.iloc[:59].drop('y', axis = 1)
    datatrainfold1 = data.iloc[59:].reset_index().drop('index', axis = 1)
    datasetfold2 = data.iloc[59:118].drop('y', axis = 1).reset_index().drop('index', axis = 1)
    datatrainfold2 = pd.concat([data.iloc[:59], data.iloc[118:]]).reset_index().drop('index', axis = 1)
    datasetfold3 = data.iloc[118:177].drop('y', axis = 1).reset_index().drop('index', axis = 1)
    datatrainfold3 = pd.concat([data.iloc[:118], data.iloc[177:]]).reset_index().drop('index', axis = 1)
    datasetfold4 = data.iloc[177:236].drop('y', axis = 1).reset_index().drop('index', axis = 1)
    datatrainfold4 = pd.concat([data.iloc[:177], data.iloc[236:]]).reset_index().drop('index', axis = 1)
    datasetfold5 = data.iloc[236:295].drop('y', axis = 1).reset_index().drop('index', axis = 1)
    datatrainfold5 = data.iloc[0:236].reset_index().drop('index', axis = 1)
    hasilfold1 = Bayes(datatrainfold1, datasetfold1)
    hasilfold2 = Bayes(datatrainfold2, datasetfold2)
    hasilfold3 = Bayes(datatrainfold3, datasetfold3)
    hasilfold4 = Bayes(datatrainfold4, datasetfold4)
    hasilfold5 = Bayes(datatrainfold5, datasetfold5)
    akurasi = []

    positive = 0
    for i in range(len(hasilfold1)):
        if hasilfold1['y'][i] == data['y'][i]: positive  += 1
    Akurasi = positive / len(hasilfold1)
    akurasi.append(Akurasi)
    
    positive  = 0
    for i in range(len(hasilfold2)):
        if hasilfold2['y'][i] == data.iloc[59:118]['y'][i+59]: positive += 1
    Akurasi = positive / len(hasilfold2)
    akurasi.append(Akurasi)
    
    positive  = 0
    for i in range(len(hasilfold3)):
        if hasilfold3['y'][i] == data.iloc[118:177]['y'][i+118]: positive += 1
    Akurasi = positive / len(hasilfold3)
    akurasi.append(Akurasi)

    positive = 0
    for i in range(len(hasilfold4)):
        if hasilfold4['y'][i] == data.iloc[177:236]['y'][i+177]: positive += 1
    Akurasi = positive / len(hasilfold4)
    akurasi.append(Akurasi)

    positive  = 0
    for i in range(len(hasilfold5)):
        if hasilfold5['y'][i] == data.iloc[236:295]['y'][i+236]: positive += 1
    Akurasi = positive / len(hasilfold5)
    akurasi.append(Akurasi)

    return akurasi
    
#=====================================================================
#                          Main Program
#=====================================================================
if __name__ == "__main__":
    start = time.time()
    print("=====================================================")
    print("                         Akurasi                     ")
    print("=====================================================")
    print("Akurasi Fold 1 dataset (0-59)   :", Evaluasi(data)[0]*100, end="%\n")
    print("Akurasi Fold 2 dataset (59-118) :", Evaluasi(data)[1]*100, end="%\n")
    print("Akurasi Fold 3 dataset (118-177):", Evaluasi(data)[2]*100, end="%\n")
    print("Akurasi Fold 4 dataset (177-236):", Evaluasi(data)[3]*100, end="%\n")
    print("Akurasi Fold 5 dataset (236-296):", Evaluasi(data)[4]*100, end="%\n")
    print("=====================================================")

    hasil = Bayes(data, dataTest)
    hasil = pd.merge(dataTest, hasil, how='inner', on ='id')
    hasil.to_excel('hasil.xlsx', index = False)
    end = time.time()
    print()
    x = input("Tekan enter untuk menyimpan hasil ke folder...")
    print()
    print("-----------------------------------------------------")
    print("Predikisi telah dibuat di file hasil.xlsx")
    print("-----------------------------------------------------")
    print("Running time:",end-start)