import matplotlib.pyplot as plt

file = open('log_20220505154422.txt')  # 打开文档
data = file.readlines()  # 读取文档数据
epoch,x,y=[1],[],[]

i=1
Loss,ACC=[],[]
loss = 0
train_acc = 0
for num in data:
	# split用于将每一行数据用逗号分割成多个对象
    #取分割后的第0列，转换成float格式后添加到para_1列表中

    if epoch.count(int(num.split(' ')[0]))>0:
        loss+=float(num.split(' ')[4])
        train_acc+=float(num.split(' ')[7][:-2])
        i+=1


    else:
        Loss.append(loss/i)
        ACC.append((train_acc/i)/100)
        i=1
        loss = 0
        train_acc = 0
        epoch.append(int(num.split(' ')[0]))


print(epoch)
print(Loss)

plt.plot(epoch[:100],Loss,color='r',label='train_loss',lw=2)
plt.plot(epoch[:100],ACC,color='g',label='train_acc',lw=2)
plt.xlabel("epoch")
plt.ylabel("acc&&loss")
plt.legend()
plt.savefig('train.png')
plt.show()

# import numpy as np
#
# a = np.array([[1, 2, 3, 4], [2, 3, 4, 5], [4, 5, 6, 7]])
# print(a)
#
# meanA_row = a.mean(axis=0)  # 计算完之后array的长度等于列数
# meanA_col = a.mean(axis=1)  # 计算完之后array的长度等于行数
#
# print(type(meanA_row))
# print(meanA_row)
# print(meanA_col)
