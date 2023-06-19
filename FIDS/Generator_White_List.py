import csv
import hashlib
import matplotlib.pyplot as plt

# f_in1 = open('C:\\Users\\da1208\\PycharmProjects\\PIDS\\soul\\train\\Pre_train_S_0.csv', 'r')
# f_out1 = open('C:\\Users\\da1208\\PycharmProjects\\PIDS\\WhiteList\\Avante_White_List_window10.csv', 'a')
# f_in1 = open('C:\\Users\\da1208\\PycharmProjects\\PIDS\\Soul\\normal_run_data.csv', 'r')
# f_out1 = open('C:\\Users\\da1208\\PycharmProjects\\PIDS\\WhiteList\\Soul_White_List_window10.csv', 'a')
# rd1 = csv.reader(f_in1)

def test1(K):
    f_in1 = open('C:\\Users\\da1208\\PycharmProjects\\PIDS\\Soul\\normal_run_data.csv', 'r')
    # f_in1 = open('test.csv', 'r')
    rd1 = csv.reader(f_in1)
    Window_Size = K
    ID_Value = []
    White_List_Temp = []
    White_List = []

    for line in rd1:
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
            m.update(bytes("".join(temp), 'utf-8'))
            White_List_Temp.append(m.hexdigest())
            temp.clear()
    window_sliding()
    White_List = list(set(White_List_Temp))
    counter = {}
    for value in White_List_Temp:
        if value in counter:
            counter[value] += 1
        else:
            counter[value] = 1


    temp = 0
    for i in range(0, len(White_List)):
        if counter[White_List[i]] > 30:
            temp += counter[White_List[i]]
    print(Window_Size, " : ",temp,"\n")
    return round((temp/len(White_List_Temp)),3)

def test2(K):
    f_in1 = open('C:\\Users\\da1208\\PycharmProjects\\PIDS\\Avante\\train\\Pre_train_S_0.csv', 'r')
    # f_in1 = open('test.csv', 'r')
    rd1 = csv.reader(f_in1)
    Window_Size = K
    ID_Value = []
    White_List_Temp = []
    White_List = []

    for line in rd1:
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
            m.update(bytes("".join(temp), 'utf-8'))
            White_List_Temp.append(m.hexdigest())
            temp.clear()

    window_sliding()
    White_List = list(set(White_List_Temp))
    counter = {}
    for value in White_List_Temp:
        if value in counter:
            counter[value] += 1
        else:
            counter[value] = 1


    temp = 0
    for i in range(0, len(White_List)):
        if counter[White_List[i]] > 30:
            temp += counter[White_List[i]]
    print(Window_Size, " : ",temp,"\n")
    return round((temp/len(White_List_Temp)),3)



# n1 = []
# n2 = []
# for i in range(1, 27):
#     n1.append(test1(i))
#
# for i in range(1, 31):
#     n2.append(test2(i))

# plt.plot(n2, 'r')
# plt.title('Avante CN7 Dataset')
# plt.xlabel('Window Size')
# plt.ylabel('Duplicated rate')
# plt.xticks(range(1, 25))
# plt.xlim(0, 25)

# plt.plot(n1)
# plt.title('Sonata RF Dataset')
# plt.xlabel('Window Size')
# plt.ylabel('Duplicated rate')
# plt.xticks(range(1, 25))
# plt.xlim(0, 25)

# plt.savefig('Sonata.png', dpi=300)
# plt.show()




