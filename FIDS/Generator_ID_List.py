import csv

f_in = open('/Users/seungmin/PycharmProjects/FIDS/DataSet/Avante/train/Pre_train_S_0.csv', 'r')
f_out = open('/Users/seungmin/PycharmProjects/FIDS/IDList/Avante_ID_List.csv', 'a')
# f_in = open('/Users/seungmin/PycharmProjects/FIDS/DataSet/Soul/normal_run_data.csv', 'r')
# f_out = open('/Users/seungmin/PycharmProjects/FIDS/IDList/Soul_ID_List.csv', 'a')
rd = csv.reader(f_in)

ID_List_Temp = []
ID_List = []

for line in rd:
    if line[1] == '0002':
        str = line[1]
        temp = bin(int(str[-1:], base=16))
        ID_List_Temp.append(temp[2:].zfill(11))
    else:
        str = line[1]
        temp = bin(int(str[-3:], base=16))
        ID_List_Temp.append(temp[2:].zfill(11))

ID_List = list(set(ID_List_Temp))

for i in range(0, len(ID_List)):
    f_out.write(ID_List[i])
    f_out.write("\n")
f_out.close()
