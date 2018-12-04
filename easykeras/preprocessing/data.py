#coding:utf-8

class DataProcessor(object):

  def get_train_examples(self, data_dir):
      raise NotImplementedError()

  def get_dev_examples(self, data_dir):
      raise NotImplementedError()

  def get_test_examples(self, data_dir):
      raise NotImplementedError()
  def get_labels(self):
      '''
      获取类别标签
      :return: 返回列表
      example:
      return ["1", "2", "3"]
      '''
      raise NotImplementedError()
  def _create_examples(self,data_dir,set_type):
      '''
      创建样本的方法
      :param data_dir: 数据目录
      :param set_type: 样本类型['train','dev','test']
      :return:
      '''
      raise NotImplementedError()