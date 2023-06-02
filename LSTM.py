import csv
import hashlib

f_in1 = open('/Users/seungmin/PycharmProjects/FIDS/LSTM_result/Avante/Cybersecurity_Car_Hacking_Prediction_window3_slide3_avante.csv', 'r') #받은 데이터셋
f_in2 = open('/Users/seungmin/PycharmProjects/FIDS/WhiteList/Avante_White_List_Window3.csv', 'r') #WhiteList
f_in3 = open('/Users/seungmin/PycharmProjects/FIDS/IDList/Avante_ID_List.csv', 'r') #ID_List
# f_in1 = open('/Users/seungmin/PycharmProjects/FIDS/LSTM_result/Soul/Gear_dataset_window10_slide10_soul.csv', 'r') #받은 데이터셋
# f_in2 = open('/Users/seungmin/PycharmProjects/FIDS/WhiteList/Soul_White_List_Window10.csv', 'r') #WhiteList
# f_in3 = open('/Users/seungmin/PycharmProjects/FIDS/IDList/Soul_ID_List.csv', 'r') #ID_List
rd1 = csv.reader(f_in1)
rd2 = csv.reader(f_in2)
rd3 = csv.reader(f_in3)

Window_Value = []
Label = []
White_List = []
ID_List = []
Window_Size = 3

for line in rd1:
    Window_Value.append(line[0])
    Label.append(line[-1])
for line in rd2:
    White_List.append(line[0])
for line in rd3:
    ID_List.append(line[0])

def white_list(str, i):
    m = hashlib.md5()
    m.update(bytes(str, 'utf-8'))
    if m.hexdigest() in White_List:
        Label[i] = 'TN'

def black_list1(str, k):
    temp = str
    ID = []
    count1 = 0
    for i in range(0, Window_Size):
        count = i * 11
        ID.append(temp[count:11+count])
    for i in range(0, Window_Size):
        if ID[i] in ID_List:
            count1 += 1
    if count1 != Window_Size:
        Label[k] = "TP"

def black_list2(str, k):
    temp = str
    ID_temp = []
    count1 = 0
    count2 = 0
    for i in range(0, Window_Size):
        count = i * 11
        ID_temp.append(temp[count:11+count])
    count1 = len(ID_temp)
    count2 = len(list(set(ID_temp)))
    if count1 != count2:
         Label[k] = "TP"


BTN = Label.count('TN')
BFP = Label.count('FP')
BTP = Label.count('TP')
BFN = Label.count('FN')


Accuracy = ((BTN+BTP)/(BTN+BFP+BFN+BTP))
Precision = (BTP) / (BTP+BFP)
Recall = (BTP) / (BTP + BFN)

F1_score = 2 * (Precision * Recall) / (Precision + Recall)

print("보정 전 Accuracy : ", round(Accuracy,3))
print("보정 전 Precision : ", round(Precision,3))
print("보정 전 Recall : ", round(Recall,3))
print("보정 전 F1_score : ", round(F1_score,3))
print("총 Window 개수 : ", len(Window_Value))
print("#실제 공격 -> 공격으로 거부 (정탐) TP : ", BTP)
print("#실제 정상 -> 정상으로 허용 (정탐) TN : ", BTN)
print("#실제 정상 -> 공격으로 거부 (오탐) FP : ", BFP)
print("#실제 공격 -> 정상으로 허용 (미탐) FN : ", BFN)

for i in range(0, len(Window_Value)):
    if Label[i] == "FP": #White List 보정
        white_list(Window_Value[i], i)
    elif Label[i] == "FN": #Black List 보정
        black_list1(Window_Value[i], i)
        black_list2(Window_Value[i], i)


TN = Label.count('TN')
FP = Label.count('FP')
TP = Label.count('TP')
FN = Label.count('FN')

Accuracy = ((TN+TP)/(TN+FP+FN+TP))
Precision = (TP) / (TP+FP)
Recall = (TP) / (TP + FN)
F1_score = 2 * (Precision * Recall) / (Precision + Recall)
print("보정 전 Accuracy : ", round(Accuracy,3))
print("보정 전 Precision : ", round(Precision,3))
print("보정 전 Recall : ", round(Recall,3))
print("보정 전 F1_score : ", round(F1_score,3))
print("총 Window 개수 : ", len(Window_Value))
print("#실제 공격 -> 공격으로 거부 (정탐) TP : ", TP)
print("#실제 정상 -> 정상으로 허용 (정탐) TN : ", TN)
print("#실제 정상 -> 공격으로 거부 (오탐) FP : ", FP)
print("#실제 공격 -> 정상으로 허용 (미탐) FN : ", FN)

print("최종 결과\n#실제 정상 -> 공격으로 거부 (오탐) FP 감소 : ", BFP - FP)
print("#실제 정상 -> 정상으로 거부 (정탐) TN 증가 : ", TN - BTN)
print("#실제 공격 -> 정상으로 허용 (미탐) FN 감소 : ", BFN - FN)
print("#실제 공격 -> 공격으로 허용 (정탐) TP 증가 : ", TP - BTP)