import csv
import hashlib

# f_in = open('/Users/seungmin/PycharmProjects/FIDS/DataSet/Soul/normal_run_data.csv', 'r')
f_in = open('/Users/seungmin/PycharmProjects/FIDS/DataSet/Avante/train/Pre_train_S_0.csv', 'r')
# f_out = open('/Users/seungmin/PycharmProjects/FIDS/WhiteList/Avante_White_List_window3.csv', 'a')

rd = csv.reader(f_in)

Window_Size = 3
ID_Value = []
Window_Value = []
White_List_Temp = []
White_List = []

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

def window_sliding():
            for count in range(0, len(ID_Value) - Window_Size + 1):
                temp = []
                for n in range(0, Window_Size):
                    temp.append(ID_Value[count + n])
                m = hashlib.md5()
                Window_Value.append(bytes("".join(temp), 'utf-8'))
                m.update(bytes("".join(temp), 'utf-8'))
                White_List_Temp.append(m.hexdigest())
                temp.clear()

Read_File()
window_sliding()
White_List = list(set(White_List_Temp))
for i in range(0,10):
      print(ID_Value[i])

for i in range(0,10):
      print(Window_Value[i])

print(len(White_List_Temp)," ")
print(len(White_List))

# for i in range(0, len(White_List)):
#     f_out.write(White_List[i])
#     f_out.write("\n")
# f_out.close()
