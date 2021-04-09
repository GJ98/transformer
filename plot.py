import matplotlib.pyplot as plt
import re

from config import result_dir

"""
ref : https://github.com/GJ98/transformer-1/blob/master/graph.py
"""

def read(name):
    f = open(name, 'r')
    file = f.read()
    file = re.sub('\\[', '', file)
    file = re.sub('\\]', '', file)
    f.close()

    return [float(i) for idx, i in enumerate(file.split(','))]

train = read(result_dir + 'train_loss.txt')
test = read(result_dir + 'test_loss.txt')

plt.plot(train, 'r', label='train')
plt.plot(test, 'b', label='validation')
plt.legend(loc='upper right')

plt.ylabel('cross entropy loss')
plt.xlabel('epoch')

plt.title('train result')
plt.show()

print(min(train))
print(min(test))