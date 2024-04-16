"""
def test(valid_queue, net, criterion):
    net.eval()
    test_loss = 0
    target_num = torch.zeros((1, 3))  # n_classes为分类任务类别数量
    predict_num = torch.zeros((1, 3))
    acc_num = torch.zeros((1, 3))

    with torch.no_grad():
        for step, (inputs, targets) in enumerate(valid_queue):
            inputs, targets = inputs.to(device), targets.to()
            outputs, _ = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)

            pre_mask = torch.zeros(outputs.size()).scatter_(1, predicted.cpu().view(-1, 1), 1.)
            predict_num += pre_mask.sum(0)  # 得到数据中每类的预测量
            tar_mask = torch.zeros(outputs.size()).scatter_(1, targets.data.cpu().view(-1, 1), 1.)
            target_num += tar_mask.sum(0)  # 得到数据中每类的数量
            acc_mask = pre_mask * tar_mask
            acc_num += acc_mask.sum(0)  # 得到各类别分类正确的样本数量

        recall = acc_num / target_num
        precision = acc_num / predict_num
        F1 = 2 * recall * precision / (recall + precision)
        accuracy = 100. * acc_num.sum(1) / target_num.sum(1)

        print('Test Acc {}, recal {}, precision {}, F1-score {}'.format(accuracy, recall, precision, F1))

    return accuracy
"""
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, \
    multilabel_confusion_matrix

np.seterr(divide='ignore', invalid='ignore')

"""
ConfusionMetric
Mertric   P    N
P        TP    FN
N        FP    TN
"""


class ClassifyMetric(object):
    def __init__(self, numClass, labels=None):
        self.labels = labels
        self.numClass = numClass
        self.confusionMatrix = np.zeros((self.numClass,) * 2)

    def genConfusionMatrix(self, y_true, y_pred):
        return confusion_matrix(y_true, y_pred, labels=self.labels)

    def addBatch(self, y_true, y_pred):
        assert np.array(y_true).shape == np.array(y_pred).shape
        self.confusionMatrix += self.genConfusionMatrix(y_true, y_pred)

    def reset(self):
        self.confusionMatrix = np.zeros((self.numClass, self.numClass))

    def accuracy(self):
        accuracy = np.diag(self.confusionMatrix).sum() / self.confusionMatrix.sum()
        return accuracy

    def precision(self):
        precision = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=0)
        return np.nan_to_num(precision)

    def recall(self):
        recall = np.diag(self.confusionMatrix) / self.confusionMatrix.sum(axis=1)
        return recall

    def f1_score(self):
        precision = self.precision()
        recall = self.recall()
        f1_score = 2 * (precision * recall) / (precision + recall)
        return np.nan_to_num(f1_score)


if __name__ == '__main__':
    y_true = ["cat", "ant", "cat", "cat", "ant", "bird"]
    y_pred = ["ant", "ant", "cat", "cat", "ant", "cat"]
    metric = ClassifyMetric(3, ["ant", "bird", "cat"])
    metric.addBatch(y_true, y_pred)
    acc = metric.accuracy()
    precision = metric.precision()
    recall = metric.recall()
    f1Score = metric.f1_score()
    print('acc is : %f' % acc)
    print('precision is :', precision)
    print('recall is :', recall)
    print('f1_score is :', f1Score)

    print('\n')
    # 与 sklearn 对比

    metric = confusion_matrix(y_true, y_pred, labels=["ant", "bird", "cat"])
    accuracy_score1 = accuracy_score(y_true, y_pred)
    print("accuracy_score", accuracy_score1)
    precision_score1 = precision_score(y_true, y_pred, average=None, zero_division=0)
    print("precision_score", precision_score1)
    recall_score1 = recall_score(y_true, y_pred, average=None, zero_division=0)
    print("recall_score", recall_score1)
    f1_score1 = f1_score(y_true, y_pred, average=None)
    print("f1_score", f1_score1)



