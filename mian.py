from keras.layers import Input, Dense
from keras.models import Model
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(1337)

from keras.datasets import mnist
import pandas as pd
import csv
import math
import random
import xlrd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from numpy import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
import numpy as np


# 定义函数
def ReadMyCsv(SaveList, fileName):
    csv_reader = csv.reader(open(fileName))
    for row in csv_reader:  # 把每个rna疾病对加入OriginalData，注意表头
        SaveList.append(row)
    return

def storFile(data, fileName):
    with open(fileName, "w", newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerows(data)
    return


# 读取源文件
OriginalData = []
ReadMyCsv(OriginalData, "drug-disease-whole.csv")
print(len(OriginalData))

# 预处理
# 小写OriginalData
counter = 0
while counter < len(OriginalData):
    OriginalData[counter][0] = OriginalData[counter][0].lower()
    OriginalData[counter][1] = OriginalData[counter][1].lower()
    counter = counter + 1
print('小写OriginalData')
LncDisease = []
counter = 0
while counter < len(OriginalData):
    Pair = []
    Pair.append(OriginalData[counter][0])
    Pair.append(OriginalData[counter][1])
    LncDisease.append(Pair)
    counter = counter + 1
storFile(LncDisease, 'LncDisease.csv')
print('LncDisease的长度', len(LncDisease))
print('OriginalData的长度', len(OriginalData))

# 构建AllDisease
AllDisease = []
counter1 = 0
while counter1 < len(OriginalData): #顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(AllDisease):  #遍历AllDisease
        if OriginalData[counter1][1] != AllDisease[counter2]:#有新疾病
            counter2 = counter2 + 1
        elif OriginalData[counter1][1] == AllDisease[counter2]:#没有新疾病，用两个if第二个if会越界
            flag = 1
            counter2 = counter2 + 1
    if flag == 0:
        AllDisease.append(OriginalData[counter1][1])
    counter1 = counter1 + 1
print('len(AllDisease)', len(AllDisease))
# storFile(AllDisease, 'AllDisease.csv')
# 构建AllDRUG
AllDRUG = []
counter1 = 0
while counter1 < len(OriginalData): #顺序遍历原始数据，构建AllDisease
    counter2 = 0
    flag = 0
    while counter2 < len(AllDRUG):  #遍历AllDisease
        if OriginalData[counter1][0] != AllDRUG[counter2]:#有新疾病
            counter2 = counter2 + 1
        elif OriginalData[counter1][0] == AllDRUG[counter2]:#没有新疾病，用两个if第二个if会越界
            flag = 1
            break
    if flag == 0:
        AllDRUG.append(OriginalData[counter1][0])
    counter1 = counter1 + 1
print('len(AllDRUG)', len(AllDRUG))
# storFile(AllDRUG,'AllDRUG.csv')
# 由drug-disease生成对应关系矩阵，有关系1，没关系0，行为疾病AllDisease，列为 AllDRUG
# 生成全0矩阵
DiseaseAndDrugBinary = []
counter = 0
while counter < len(AllDisease):
    row = []
    counter1 = 0
    while counter1 < len(AllDRUG):
        row.append(0)
        counter1 = counter1 + 1
    DiseaseAndDrugBinary.append(row)
    counter = counter + 1


print('len(LncDisease)', len(LncDisease))
counter = 0
while counter < len(LncDisease):
    DN = LncDisease[counter][1]
    RN = LncDisease[counter][0]
    counter1 = 0
    while counter1 < len(AllDisease):
        if AllDisease[counter1] == DN:
            counter2 = 0
            while counter2 < len(AllDRUG):
                if AllDRUG[counter2] == RN:
                    DiseaseAndDrugBinary[counter1][counter2] = 1
                    break
                counter2 = counter2 + 1
            break
        counter1 = counter1 + 1
    counter = counter + 1
print('len(DiseaseAndDrugBinary)', len(DiseaseAndDrugBinary))
# storFile(DiseaseAndDrugBinary, 'DiseaseAndDrugBinary.csv')
# disease的文本挖掘相似矩阵
lines = [line.strip().split() for line in open("disease相似性矩阵.txt")]
txtSimilarity = []
i = 0
for dis in lines:
    i = i + 1
    if i == 1:
        continue
    txtSimilarity.append(dis[1:])
print('len(txtSimilarity)',len(txtSimilarity))
print('len(txtSimilarity[1])',len(txtSimilarity[1]))

# drug的文本挖掘相似矩阵
lines = [line.strip().split() for line in open("drug相似性矩阵.txt")]
drugtxtSimilarity = []
i = 0
for dis in lines:
    i = i + 1
    if i == 1:
        continue
    drugtxtSimilarity.append(dis[1:])
print('len(drugtxtSimilarity)',len(drugtxtSimilarity))
print('len(drugtxtSimilarity[1])',len(drugtxtSimilarity[1]))
# 计算rd
counter1 = 0
sum1 = 0
while counter1 < (len(AllDisease)):
    counter2 = 0
    while counter2 < (len(AllDRUG)):
        sum1 = sum1 + pow((DiseaseAndDrugBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nd = len(AllDisease)
rdpie = 0.5
rd = rdpie * Nd / Ak
print('disease rd', rd)
# 生成DiseaseGaussian
DiseaseGaussian = []
counter1 = 0
while counter1 < len(AllDisease):#计算疾病counter1和counter2之间的similarity
    counter2 = 0
    DiseaseGaussianRow = []
    while counter2 < len(AllDisease):# 计算Ai*和Bj*
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllDRUG):#疾病的每个属性分量
            sum2 = pow((DiseaseAndDrugBinary[counter1][counter3] - DiseaseAndDrugBinary[counter2][counter3]), 2)#计算平方
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj/rd))
        DiseaseGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    DiseaseGaussian.append(DiseaseGaussianRow)
    counter1 = counter1 + 1
print('len(DiseaseGaussian)', len(DiseaseGaussian))
print('len(DiseaseGaussian[0])', len(DiseaseGaussian[0]))
# 构建Drugaussian
from numpy import *
MDiseaseAndDrugBinary = np.array(DiseaseAndDrugBinary)    # 列表转为矩阵
DRUGAndDiseaseBinary = MDiseaseAndDrugBinary.T    # 转置DiseaseAndMiRNABinary
DRUGGaussian = []
counter1 = 0
sum1 = 0
while counter1 < (len(AllDRUG)):     # rna数量
    counter2 = 0
    while counter2 < (len(AllDisease)):     # disease数量
        sum1 = sum1 + pow((DRUGAndDiseaseBinary[counter1][counter2]), 2)
        counter2 = counter2 + 1
    counter1 = counter1 + 1
print('sum1=', sum1)
Ak = sum1
Nm = len(AllDRUG)
rdpie = 0.5
rd = rdpie * Nm / Ak
print('DRUG rd', rd)
# 生成DRUGGaussian
counter1 = 0
while counter1 < len(AllDRUG):   # 计算rna counter1和counter2之间的similarity
    counter2 = 0
    DRUGGaussianRow = []
    while counter2 < len(AllDRUG):   # 计算Ai*和Bj*
        AiMinusBj = 0
        sum2 = 0
        counter3 = 0
        AsimilarityB = 0
        while counter3 < len(AllDisease):   # rna的每个属性分量
            sum2 = pow((DRUGAndDiseaseBinary[counter1][counter3] - DRUGAndDiseaseBinary[counter2][counter3]), 2)#计算平方，有问题？？？？？
            AiMinusBj = AiMinusBj + sum2
            counter3 = counter3 + 1
        AsimilarityB = math.exp(- (AiMinusBj/rd))
        DRUGGaussianRow.append(AsimilarityB)
        counter2 = counter2 + 1
    DRUGGaussian.append(DRUGGaussianRow)
    counter1 = counter1 + 1
print('type(DRUGGaussian)', type(DRUGGaussian))
print('len(DRUGGaussian)', len(DRUGGaussian))
print('len(DRUGGaussian[0])', len(DRUGGaussian[0]))
# 挑选正负样本
# 挑选正/负例
import random
counter1 = 0    # 在疾病中随机选择
counter2 = 0    # 在rna中随机选择
counterP = 0    # 正样本数量
counterN = 0    # 负样本数量
PositiveSample = []     # rna - disease 对
# 若正例为全部的RNA-Disease对
PositiveSample = LncDisease
print('PositiveSample)', len(PositiveSample))

# storFile(PositiveSample, 'PositiveSample.csv')


# 负样本为全部的disease-drug（313*593）中随机抽取，未在内LncDisease即为负样本
NegativeSample = []
counterN = 0
while counterN < len(PositiveSample):                         # 当正负样本任一小于10时执行循环，10用来测试，应与正样本数目相同，len(PositiveSample)！！！！！！！！！！！！！！！！！！！！
    counterD = random.randint(0, len(AllDisease)-1)
    counterR = random.randint(0, len(AllDRUG)-1)     # 随机选出一个疾病rna对
    DiseaseAndRnaPair = []
    DiseaseAndRnaPair.append(AllDRUG[counterR])
    DiseaseAndRnaPair.append(AllDisease[counterD])
    flag1 = 0
    counter = 0
    while counter < len(LncDisease):
        if DiseaseAndRnaPair == LncDisease[counter]:
            flag1 = 1
            break
        counter = counter + 1
    if flag1 == 1:
        continue
    flag2 = 0
    counter1 = 0
    while counter1 < len(NegativeSample):
        if DiseaseAndRnaPair == NegativeSample[counter1]:
            flag2 = 1
            break
        counter1 = counter1 + 1
    if flag2 == 1:
        continue
    if (flag1 == 0 & flag2 == 0):
        NegativePair = []
        NegativePair.append(AllDRUG[counterR])
        NegativePair.append(AllDisease[counterD])
        NegativeSample.append(NegativePair)
        counterN = counterN + 1
print('len(NegativeSample)', len(NegativeSample))
# 由txtSimilarity，Gaussian生成最终的Similarity，有语义相似性，在model1/2矩阵中有值的就用model，没有的就用高斯，合成一个矩阵
DiseaseSimilarity = []
counter = 0
while counter < len(AllDisease):
    counter1 = 0
    Row = []
    while counter1 < len(AllDisease):
        v = float(DiseaseGaussian[counter][counter1])
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(txtSimilarity[counter][counter1])
        counter1 = counter1 + 1
    DiseaseSimilarity.append(Row)
    counter = counter + 1
print('len(DiseaseSimilarity)', len(DiseaseSimilarity))
print('len(DiseaseSimilarity[0])',len(DiseaseSimilarity[0]))
# storFile(DiseaseSimilarity, 'DiseaseSimilarity.csv')

DRUGSimilarity = []
counter = 0
while counter < len(AllDRUG):
    counter1 = 0
    Row = []
    while counter1 < len(AllDRUG):
        v = float(DRUGGaussian[counter][counter1])
        if v > 0:
            Row.append(v)
        if v == 0:
            Row.append(drugtxtSimilarity[counter][counter1])
        counter1 = counter1 + 1
    DRUGSimilarity.append(Row)
    counter = counter + 1
print('len(DRUGSimilarity)', len(DRUGSimilarity))
print('len(DRUGSimilarity[0)',len(DRUGSimilarity[0]))
# storFile(DRUGSimilarity, 'DRUGSimilarity.csv')

# 生成训练集 ！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！！打乱顺序？？？
AllSample = PositiveSample.copy()
AllSample.extend(NegativeSample)        # drug-disease

# SampleFeature
SampleFeature = []
counter = 0
while counter < len(AllSample):
    counter1 = 0
    while counter1 < len(AllDRUG):
        if AllSample[counter][0] == AllDRUG[counter1]:
            a = []
            counter3 = 0
            # 原本是ALLDrug
            while counter3 <len(DRUGSimilarity[0]):
                v = DRUGSimilarity[counter1][counter3]
                a.append(v)
                counter3 = counter3 + 1
            break
        counter1 = counter1 + 1
    counter2 = 0
    while counter2 < len(AllDisease):
        if AllSample[counter][1] == AllDisease[counter2]:
            b = []
            counter3 = 0
            # 原本是ALLDisease
            while counter3 < len(DiseaseSimilarity[0]):
                v = DiseaseSimilarity[counter2][counter3]
                b.append(v)
                counter3 = counter3 + 1
            break
        counter2 = counter2 + 1
    a.extend(b)
    SampleFeature.append(a)
    counter = counter + 1
counter1 = 0
storFile(SampleFeature, 'SampleFeature.csv')
print('SampleFeature',len(SampleFeature))
print('SampleFeature[1]',len(SampleFeature[1]))






