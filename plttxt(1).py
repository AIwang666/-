import matplotlib.pyplot as plt

file = open('1.txt')  # 打开文档
data = file.readlines()  # 读取文档数据
x,y,loss=[],[],[]

for num in data:
	# split用于将每一行数据用逗号分割成多个对象
    #取分割后的第0列，转换成float格式后添加到para_1列表中
    print(num.split(',')[1][-7:-2])
    x.append(int(num.split(',')[0][-3:]))
    y.append(float(num.split(',')[1][-7:-2])/100)
    loss.append(float(num.split(',')[2][-6:]))

plt.plot(x,y,color='r',lw=2,label='test_acc')
plt.plot(x,loss,color='g',lw=2,label='test_loss')
plt.xlabel("epoch")
plt.ylabel("acc && loss")
plt.title("test acc and loss")
plt.legend()
plt.savefig('test.png')
plt.show()

