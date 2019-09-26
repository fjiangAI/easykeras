# coding:utf-8

def read_data_file(data_file, split=' ', num=None, mode='text', encoding='utf-8'):
    """读取数据文件

    数据文件每行一个样例，格式为:
        item_1[split]item_2[split]...[split]item_n
    [split]为分隔符
    
    Args:
        data_file: 数据文件路径
        split: 分隔符，默认为空格
        num: 要读取的数据列数，默认读取所有列
        mode: 读文件模式，text或bin，默认text
        encoding: 文件编码，默认utf-8

    Returns:
        一个或多个样例列表，每个列表对应一列

        item_1, item_2 = read_data_file('data.txt', num=2)
    """
    with open(data_file, mode=('rt' if mode == 'text' else 'rb'), encoding=encoding) as fin:
        n = len(fin.readline().strip().split(split))
        if n == 0:
            print('读取文件失败，请检查!')
            return None
        fin.seek(0)
        if not num:
            num = n
        elif num > n:
            num = n
        if num == 1:
            return [line.strip().split(split)[0] for line in fin]
        result_list = [[] for i in range(num)]
        for line in fin:
            for i,item in enumerate(line.strip().split(split)[:num]):
                result_list[i].append(item)
        return tuple(result_list)


class DataProcessor(object):
    """
    数据处理基类，在具体任务的时候再实现数据的格式化
    """
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