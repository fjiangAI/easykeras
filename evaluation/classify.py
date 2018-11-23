__author__ = 'jf'
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
def cal_PRF1(y_true,y_pred,average='micro'):
    '''
    全局计算PRF1值
    :param y_true:
    真实标签
    :param y_pred:
    预测标签
    :param average:
    可选有 None,'micro','macro','weighted'，'samples'
    :return: 返回(p,r,f)
    '''
    precision=metrics.precision_score(y_true, y_pred, average=average)
    recall=metrics.recall_score(y_true, y_pred, average=average)
    f1=metrics.f1_score(y_true, y_pred, average=average)
    return precision,recall,f1
def cal_confusion_matrix(y_true,y_pred,labels=None):
    '''
    计算混淆矩阵
    :param y_true: 真实标签
    :param y_pred: 预测标签
    :param labels: 按类别排序
    :return:
        预测标签
    真
    实
    标
    签
    '''
    return confusion_matrix(y_true, y_pred,labels)
def report_detail(y_true,y_pred,target_names,digits=4):
    '''
    给出各分类的结果及整体平均值，其中整体平均值使用weight的计算方法
    :param y_true:真实标签
    :param y_pred:预测标签
    :param target_names: 类别
    :param digits: 精度，默认小数点后4位
    :return: 统计文本（字符串）
    '''
    return classification_report(y_true, y_pred,target_names=target_names,digits=digits)