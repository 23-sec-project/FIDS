import csv
import hashlib

f_in = open('/Users/seungmin/PycharmProjects/FIDS/DataSet/Avante/test/Pre_submit_SD.csv', 'r') #분석 데이터
# f_in = open('/Users/seungmin/PycharmProjects/FIDS/DataSet/Avante/test/Pre_submit_D.csv', 'r') #분석 데이터


rd = csv.reader(f_in)

Window_Size = 3
ID_Value = []
Window_Value = []

def Read_File():
    for line in rd:
            if line[1] == '0002':
                str = line[1]
                temp = bin(int(str[-1:], base=16))
                ID_Value.append(temp[2:].zfill(11))
            else:
                str = line[1]
                temp = bin(int(str[-3:], base=16))
                ID_Value.append(temp[2:].zfill(11))

def window_by_window():
    count = 0
    val = divmod(len(ID_Value), Window_Size)
    while (count < val[0] * Window_Size):
        temp = []
        for n in range(0, Window_Size):
            temp.append(ID_Value[count + n])
        Window_Value.append(bytes("".join(temp), 'utf-8'))
        temp.clear()
        count += Window_Size

Read_File()
window_by_window()

print(len(Window_Value))

# 기존 CSV 파일 읽어오기
with open('/Users/seungmin/PycharmProjects/FIDS/LSTM_result/avantee/Cybersecurity_Car_Hacking_Prediction_window10_slide10_avante.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)

# 데이터 수정하여 내용 추가하기
for i in range(0, len(data)):
     data[i][0] = Window_Value[i].decode('utf-8')

# 수정된 데이터를 새로운 CSV 파일에 쓰기
with open('/Users/seungmin/PycharmProjects/FIDS/LSTM_result/Avante/Cybersecurity_Car_Hacking_Prediction_window10_slide10_avante.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
