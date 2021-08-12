import os, shutil
datalist = "./data_sets/cat_12/train_list.txt"
base_dir = './data_sets/cat_12/cat_12_train'
# target_train_dir = 'target_train_data_sets'
# target_validation_dir = 'target_validation_data_sets'
# all_data_dir = 'target_all_data_sets'
target_validation_dir = 'target_K_validation_data_sets' # 交叉验证
target_train_dir = 'target_K_train_data_sets'
# for(i) in range(12): ## 建立训练集
#     train_dir = os.path.join(all_data_dir, str(i))
#     if not os.path.exists(train_dir):
#         os.makedirs(train_dir)

for(i) in range(12): ## 建立训练集
    train_dir = os.path.join(target_train_dir, str(i))
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

for(i) in range(12): ## 建立验证集
    train_dir = os.path.join(target_validation_dir, str(i))
    if not os.path.exists(train_dir):
        os.makedirs(train_dir)


with open(datalist, "r") as f:
    data = f.readlines()
    for(i)in range(len(data)):
        data[i] = data[i].split()
        fname = data[i][0].split('/')
        print(fname[1])
        print(data[i][1])
        src = os.path.join(base_dir, fname[1])
        if(i % 180 < 40): #每隔180张图片划分训练测试集
            dst = os.path.join('target_K_validation_data_sets/{}'.format(data[i][1]), fname[1])
        else:
            dst = os.path.join('target_K_train_data_sets/{}'.format(data[i][1]), fname[1])
        shutil.copyfile(src, dst)
    # print(data)
