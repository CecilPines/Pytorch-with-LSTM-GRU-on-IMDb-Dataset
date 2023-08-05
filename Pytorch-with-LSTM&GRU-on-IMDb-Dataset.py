# 库模块导入
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import re
import os
from tqdm import tqdm

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

MAX_WORD = 10000  # 只保留最高频的10000词
MAX_LEN = 300     # 句子统一长度为300
word_count={}     # 词典，统计词出现的词数


#清理文本，去标点符号，转小写
def clean_str(string):
    string = re.sub(r"[^A-Za-z0-9]", " ", string)
    string = re.sub(r"<br", " ", string)
    string = re.sub(r"/>", " ", string)
    string = re.sub(r"\'", " \'", string)

    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

# 分词方法
def tokenizer(sentence):
    return sentence.split()

#  数据预处理过程
def data_process(text_path, text_dir): # 根据文本路径生成文本的标签

    print("data preprocess")
    # 打开新文档，预处理后text添加入该文档
    file_pro = open(text_path,'w',encoding='utf-8')

    file_tag = ['pos', 'neg'] # 只使用pos和neg中文件
    for tag in file_tag:

      f_path = os.path.join(text_dir, tag)

      # 得到Label
      if tag == 'pos':
        label = '1'
      elif tag == 'neg':
        label = '0'

      # 遍历文件夹中文件
      for file_name in os.listdir(f_path):
        # 判断是否为txt文件
        if not file_name.endswith('txt'):
          continue
        file_path = os.path.join(f_path, file_name)
        # 打开文本
        f = open(file_path, 'r', encoding='utf-8')
        # 清理文本
        clean_text = clean_str(f.readline())
        # 分割文本
        tokens = clean_text.split()
        # 统计词频
        for token in tokens:
          if token in word_count.keys():
            word_count[token] += 1
          else:
            word_count[token] = 1
        file_pro.write(label + ' ' + clean_text +'\n')
        f.close()
        file_pro.flush()

    file_pro.close()
    print("build vocabulary")

    vocab = {"<UNK>": 0, "<PAD>": 1}

    word_count_sort = sorted(word_count.items(), key=lambda item : item[1], reverse=True) # 对词进行排序，过滤低频词，只取前MAX_WORD个高频词
    word_number = 1
    for word in word_count_sort:
        if word[0] not in vocab.keys():
            vocab[word[0]] = len(vocab)
            word_number += 1
        if word_number > MAX_WORD:
            break
    return vocab

# 定义Dataset
class MyDataset(Dataset):
    def __init__(self, text_path):
      file = open(text_path, 'r', encoding='utf-8')
      self.text_with_tag = file.readlines()  # 文本标签与内容
      file.close()

    # 重写getitem
    def __getitem__(self, index):
      # 获取一个样本的标签和文本信息
      line = self.text_with_tag[index]
      label = int(line[0]) # 标签信息
      text = line[2:-1]  # 文本信息
      return text, label

    def __len__(self):
      return len(self.text_with_tag)
    
# 根据vocab将句子转为定长MAX_LEN的tensor
def text_transform(sentence_list, vocab):
    sentence_index_list = []
    for sentence in sentence_list:
        sentence_idx = [vocab[token] if token in vocab.keys() else vocab['<UNK>'] for token in tokenizer(sentence)] # 句子分词转为id

        if len(sentence_idx) < MAX_LEN:
            for i in range(MAX_LEN-len(sentence_idx)): # 对长度不够的句子进行PAD填充
                sentence_idx.append(vocab['<PAD>'])

        sentence_idx = sentence_idx[:MAX_LEN] # 取前MAX_LEN长度
        sentence_index_list.append(sentence_idx)
    return torch.LongTensor(sentence_index_list) # 将转为idx的词转为tensor

# 定义LSTM模型
class LSTM(nn.Module):
    def __init__(self, vocab, input_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(len(vocab), input_size)
        # LSTM层
        self.lstm = nn.LSTM(input_size=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   bidirectional=False)
        # 全连接层
        self.fc = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1,0)) # permute(1,0)交换维度
        outputs, _ = self.lstm(embeddings)
        encoding = outputs[-1]
        outs = self.softmax(self.fc(encoding)) # 输出层为二维概率[a,b]
        return outs

# 定义GRU模型
class GRU(nn.Module):
    def __init__(self, vocab, input_size, hidden_size, num_layers):
        super(GRU, self).__init__()
        # embedding层
        self.embedding = nn.Embedding(len(vocab), input_size)
        # LSTM层
        self.lstm = nn.GRU(input_size=input_size,
                   hidden_size=hidden_size,
                   num_layers=num_layers,
                   bidirectional=False)
        # 全连接层
        self.fc = nn.Linear(hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, inputs):
        embeddings = self.embedding(inputs.permute(1,0)) # permute(1,0)交换维度
        outputs, _ = self.lstm(embeddings)
        encoding = outputs[-1]
        outs = self.softmax(self.fc(encoding)) # 输出层为二维概率[a,b]
        return outs

# 定义绘制loss曲线函数
def DrawLoss(file_name, label_name):

    with open(file_name, 'r') as f:
        raw_data = f.read()
        # [-1:1]是为了去除文件中的前后中括号"[]"
        data = raw_data[1:-1].split(",")

    y_loss = np.asfarray(data, float)
    x_loss = range(len(y_loss))
    plt.figure()

    # 去除顶部和右侧边框
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.xlabel('iters')    # x轴标签
    plt.ylabel('loss')     # y轴标签

    # 以x_loss为横坐标，y_loss为纵坐标，曲线宽度为1，实线，增加标签，训练损失，
    # 默认颜色，如果想更改颜色，可以增加参数color='red',这是红色。
    plt.plot(x_loss, y_loss, linewidth=1, linestyle="solid", label=label_name)
    plt.legend()
    plt.title('Loss curve')
    plt.show()

# 模型训练
def train(model, train_data, vocab, epoch=10):
    # 记录训练loss
    train_losses = []

    print('train model')
    model = model.to(device)
    loss_sigma = 0.0
    correct = 0.0
    # 定义损失函数和优化器
    criterion = torch.nn.NLLLoss()
    learning_rate = 0.005
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    for epoch in tqdm(range(epoch)):
        model.train()
        avg_loss = 0  # 平均损失
        avg_acc = 0  # 平均准确率
        for idx, (text, label) in enumerate(tqdm(train_data)):

            train_x = text_transform(text, vocab).to(device)
            train_y = label.to(device)

            optimizer.zero_grad()
            pred = model(train_x)
            loss = criterion(pred.log(), train_y)
            loss.backward()
            optimizer.step()
            avg_loss += loss.item()
            avg_acc += accuracy(pred, train_y)

            train_losses.append(loss.item())

        # 一个epoch结束后，计算平均loss和评平均acc
        avg_loss = avg_loss / len(train_data)
        avg_acc = avg_acc / len(train_data)

        print("avg_loss:", avg_loss, " train_avg_acc:,", avg_acc)

        # 保存训练完成后的模型参数
        torch.save(model.state_dict(), 'IMDB_parameter.pkl')

    # 绘制train loss曲线
    file_name_train = 'train_logs.txt'
    with open(file_name_train,'w') as train_los:
        train_los.write(str(train_losses))
    DrawLoss(file_name_train, "train_loss")

# 计算预测准确性
def accuracy(y_pred, y_true):
    label_pred = y_pred.max(dim=1)[1]
    acc = len(y_pred) - torch.sum(torch.abs(label_pred-y_true)) # 正确的个数
    return acc.detach().cpu().numpy() / len(y_pred)

# 模型测试
def test(model, test_data, vocab):
    # 记录测试loss
    test_losses = []

    # 在每个类别上的正确数
    true = [0,0] # (pos, neg)
    false = [0,0]

    print('test model')
    model = model.to(device)
    model.eval()
    criterion = torch.nn.NLLLoss()
    avg_acc = 0
    for idx, (text, label) in enumerate(tqdm(test_data)):
        train_x = text_transform(text, vocab).to(device)
        train_y = label.to(device)
        pred = model(train_x)
        loss = criterion(pred.log(), train_y)
        test_losses.append(loss.item())

        avg_acc += accuracy(pred, train_y)

        pred_label = pred.max(dim=1)[1]
        # 预测为pos且真实为pos
        true[0] += torch.sum((pred_label == 1) & (train_y == 1)).item()
        # 预测为pos但真实为neg
        false[0] += torch.sum((pred_label == 1) & (train_y == 0)).item()
        # 预测为neg且真实为neg
        true[1] += torch.sum((pred_label == 0) & (train_y == 0)).item()
        # 预测为neg但真实为pos
        false[1] += torch.sum((pred_label == 0) & (train_y == 1)).item()

    # 绘制test loss曲线
    file_name_test = 'test_logs.txt'
    with open(file_name_test,'w') as test_los:
        test_los.write(str(test_losses))
    DrawLoss(file_name_test, "test_loss")

    # 总精度
    avg_accuracy = (true[0] + true[1]) / (false[0] + false[1] + true[0] + true[1])
    # 各类别精度
    pos_accuracy = true[0] / (true[0] + false[0])  # Adding a small value to avoid division by zero
    neg_accuracy = true[1] / (true[1] + false[1])

    # 整体精度
    print(f'Accuracy of all: {100 * avg_accuracy} %')
    # 各类别精度
    print(f'Accuracy of pos: {100 * pos_accuracy} %')
    print(f'Accuracy of neg: {100 * neg_accuracy} %')

    return avg_acc

# 主函数

def main():

    # 加载云端硬盘
    from google.colab import drive
    drive.mount('/content/drive')

    train_dir = '/content/drive/MyDrive/Colab Notebooks/aclImdb_v1/aclImdb/train'  # 原训练集文件地址
    train_path = '/content/drive/MyDrive/Colab Notebooks/train.txt'  # 预处理后的训练集文件地址

    test_dir = '/content/drive/MyDrive/Colab Notebooks/aclImdb_v1/aclImdb/test'  # 原训练集文件地址
    test_path = '/content/drive/MyDrive/Colab Notebooks/test.txt'  # 预处理后的训练集文件地址

    vocab = data_process(train_path, train_dir) # 数据预处理
    data_process(test_path, test_dir)

    # 词典保存为本地
    np.save('vocab.npy', vocab)
    # 加载本地已经存储的vocab
    vocab = np.load('vocab.npy', allow_pickle=True).item()

    # 构建MyDataset实例
    train_data = MyDataset(text_path=train_path)
    test_data = MyDataset(text_path=test_path)

    # 构建DataLoder
    train_loader = DataLoader(dataset=train_data, batch_size=256, shuffle=True)
    test_loader = DataLoader(dataset=test_data, batch_size=64, shuffle=False)

    # 生成模型
    # model = LSTM(vocab=vocab, input_size=300, hidden_size=128, num_layers=2)  # 定义LSTM模型
    model = GRU(vocab=vocab, input_size=300, hidden_size=128, num_layers=2)  # 定义GRU模型

    train(model=model, train_data=train_loader, vocab=vocab, epoch=50)

    # 加载训练好的模型
    model.load_state_dict(torch.load('IMDB_parameter.pkl', map_location=torch.device('cpu')))

    # 测试结果
    acc = test(model=model, test_data=test_loader, vocab=vocab)
    print(acc)

if __name__ == '__main__':
    main()

