"""
主要解决了`num_workers > 0`时的问题，包括：
1. MultiTFRecordDataset数据重复问题，由于每个worker都会有一个不同的dataset对象副本，因此每个worker会重复读取iterator的数据。其实TFRecordDataset已经解决了这个问题，本人按照同样的思路，在MultiTFRecordDataset中实现；
2. 所有worker会平分所有数据，因此每个worker都会出现低于batch_size的情况，`drop_last=True`会过滤`num_workers`次。通过增加`batch_size`参数，优化数据分配，实现了有且仅有一个worker会出现低于batch_size的情况；
3. 这些实现依赖index文件
"""


"""
wwm: https://github.com/ymcui/Chinese-BERT-wwm/issues/4

MacBERT: 相似的词长度不同怎么处理?
【相关解决方案】
（1）在候选列表中找不到等长相似词的时候会替换为随机token。–也是原作者的做法 但考虑到随机词的比例可能过大
（2）设置模型 在面临相似词跟原词长度不一致的问题，是跳转到下一个长度一致的相似词来解决的。-- 但训练时间太久了
（2）直接自己创造一个相似词和原词长度都一致的候选列表-我选择了第二种，具体方法是 用自己的大概20G的语料库来训练一个词向量，再提取相似词
————————————————
版权声明：本文为CSDN博主「jianafeng」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/Jiana_Feng/article/details/123539083

SOP task: 50%调换两个句子的顺序，50%保持不变

提前将词表的所有词进行tokenize编码
"""


# tokenizer = AutoTokenizer.from_pretrained('bert-base-chinese', cache_dir='../cache/')
# tokens = tokenizer.encode_plus('我们 是[MASK]这样的a nearby人')
# tokenizer.convert_tokens_to_ids(tokens)
#
# # text = '提示：安装后初次使用会下载词向量文件，下载速度取决于网络情况。'
# text = '我们 是[MASK]这样的a nearby人'
# result = tokenizer(text, max_length=32, padding=True, truncation='longest_first')
# print(result)
#
# encodings = tokenizer(text, padding=False, add_special_tokens=False)._encodings[0]
# print(encodings.tokens)
# print(len(encodings))
# # encodings = tokenizer(encodings.tokens, padding=False, add_special_tokens=False, is_split_into_words=True)._encodings[0]
#
# # cand_indexes, cand_words = get_masked_cand_indexes_wwm(text, encodings.offsets)
# # for i in range(len(cand_indexes)):
# #     print(i, [encodings.tokens[idx] for idx in cand_indexes[i]], cand_words[i])
#
# encodings_1 = Encodings(encodings, text)
# encodings_2 = Encodings(encodings, text)
# encodings_1.add_special_tokens(tokenizer, encodings_2)
# print(encodings_1)
# cand_indexes, cand_words = get_masked_cand_indexes_wwm(encodings_1)
# for i in range(len(cand_indexes)):
#     print(i, [encodings_1.tokens[idx] for idx in cand_indexes[i]], cand_words[i])

import typing
import numpy as np
import functools
import math

import torch.utils.data

from tfrecord import TFRecordWriter
from tfrecord.reader import tfrecord_loader
from tfrecord import iterator_utils

from tfrecord.torch.dataset import MultiTFRecordDataset, TFRecordDataset
from tfrecord.tools import create_index


def multi_tfrecord_loader(data_pattern: str,
                          index_pattern: typing.Union[str, None],
                          splits: typing.Dict[str, float],
                          description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                          shard: typing.Optional[typing.Tuple[int, int]] = None,
                          sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                          compression_type: typing.Optional[str] = None,
                          infinite: bool = True,
                          ) -> typing.Iterable[typing.Union[typing.Dict[str, np.ndarray],
                                                            typing.Tuple[typing.Dict[str, np.ndarray],
                                                                         typing.Dict[str, typing.List[np.ndarray]]]]]:
    """Create an iterator by reading and merging multiple tfrecord datasets.

    NOTE: Sharding is currently unavailable for the multi tfrecord loader.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str or None
        Input index path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    infinite: bool, optional, default=True
        Whether the returned iterator should be infinite or not

    Returns:
    --------
    it: iterator
        A repeating iterator that generates batches of data.
    """
    loaders = [functools.partial(tfrecord_loader, data_path=data_pattern.format(split),
                                 index_path=index_pattern.format(split) \
                                     if index_pattern is not None else None,
                                 description=description,
                                 sequence_description=sequence_description,
                                 compression_type=compression_type,
                                 shard=shard
                                 )
               for split in splits.keys()]
    return iterator_utils.sample_iterators(loaders, list(splits.values()), infinite=infinite)


class ConcurrencyMultiTFRecordDataset(torch.utils.data.IterableDataset):
    """Parse multiple (generic) TFRecords datasets into an `IterableDataset`
    object, which contain `np.ndarrays`s.

    Params:
    -------
    data_pattern: str
        Input data path pattern.

    index_pattern: str or None
        Input index path pattern.

    splits: dict
        Dictionary of (key, value) pairs, where the key is used to
        construct the data and index path(s) and the value determines
        the contribution of each split to the batch.

    description: list or dict of str, optional, default=None
        List of keys or dict of (key, value) pairs to extract from each
        record. The keys represent the name of the features and the
        values ("byte", "float", or "int") correspond to the data type.
        If dtypes are provided, then they are verified against the
        inferred type for compatibility purposes. If None (default),
        then all features contained in the file are extracted.

    shuffle_queue_size: int, optional, default=None
        Length of buffer. Determines how many records are queued to
        sample from.

    transform : a callable, default = None
        A function that takes in the input `features` i.e the dict
        provided in the description, transforms it and returns a
        desirable output.

    sequence_description: list or dict of str, optional, default=None
        Similar to `description`, but refers to the sequence features
        within a `SequenceExample`. When this field is `None`, then it
        is assumed that an `Example` is being read otherwise, a
        `SequenceExample` is read. If an empty list or dictionary is
        passed, then all features contained in the file are extracted.

    compression_type: str, optional, default=None
        The type of compression used for the tfrecord. Choose either
        'gzip' or None.

    infinite: bool, optional, default=True
        Whether the Dataset should be infinite or not
    """

    def __init__(self,
                 data_pattern: str,
                 index_pattern: typing.Union[str, None],
                 splits: typing.Dict[str, float],
                 description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 shuffle_queue_size: typing.Optional[int] = None,
                 transform: typing.Callable[[dict], typing.Any] = None,
                 sequence_description: typing.Union[typing.List[str], typing.Dict[str, str], None] = None,
                 compression_type: typing.Optional[str] = None,
                 infinite: bool = True
                 ) -> None:
        super(ConcurrencyMultiTFRecordDataset, self).__init__()
        self.data_pattern = data_pattern
        self.index_pattern = index_pattern
        self.splits = splits
        self.description = description
        self.sequence_description = sequence_description
        self.shuffle_queue_size = shuffle_queue_size
        self.transform = transform
        self.compression_type = compression_type
        self.infinite = infinite

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            print(worker_info)
            shard = worker_info.id, worker_info.num_workers
            np.random.seed(worker_info.seed % np.iinfo(np.uint32).max)
        else:
            shard = None
        it = multi_tfrecord_loader(data_pattern=self.data_pattern,
                                   index_pattern=self.index_pattern,
                                   splits=self.splits,
                                   description=self.description,
                                   shard=shard,
                                   sequence_description=self.sequence_description,
                                   compression_type=self.compression_type,
                                   infinite=self.infinite,
                                   )
        if self.shuffle_queue_size:
            it = iterator_utils.shuffle_iterator(it, self.shuffle_queue_size)
        if self.transform:
            it = map(self.transform(worker_info), it)
        return it


def _add_worker_info(worker_info):
    def _func(x):
        if worker_info is not None:
            x['worker_id'] = worker_info.id
            return x
    return _func


def _map(x):
    worker_info = torch.utils.data.get_worker_info()
    # print(worker_info)
    x['worker_id'] = worker_info.id
    return x

def write():
    writer = TFRecordWriter("../data1.tfrecord")
    for idx in range(100):
        writer.write({'idx': (idx, 'int'),
                      'inputs_id': (np.random.randint(0, 1000, [8]), 'int')})

    writer.close()

    create_index("../data1.tfrecord", "../data1.index")

    writer = TFRecordWriter("../data2.tfrecord")
    for idx in range(100, 200):
        writer.write({'idx': (idx, 'int'),
                      'inputs_id': (np.random.randint(0, 1000, [8]), 'int')})

    writer.close()

    create_index("../data2.tfrecord", "../data2.index")


class TestDataset(torch.utils.data.Dataset):
    def __init__(self):
        super(TestDataset, self).__init__()

    def __len__(self):
        return 200

    def __getitem__(self, idx):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            print(worker_info)
        return {'idx': torch.LongTensor([idx]),
                'inputs_id': torch.randint(0, 1000, [8]),
                'worker_id': worker_info.id}


class MyIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, start=0, end=200):
        super(MyIterableDataset).__init__()
        assert end > start, "this example code only works with end >= start"
        self.start = start
        self.end = end

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is None:  # single-process data loading, return the full iterator
            iter_start = self.start
            iter_end = self.end
        else:  # in a worker process
            # split workload
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        # return iter(range(iter_start, iter_end))
        return iter([{'idx': i} for i in range(iter_start, iter_end)])


def read():
    tfrecord_pattern = "../{}.tfrecord"
    index_pattern = "../{}.index"
    splits = {
        "data1": 1.,
        "data2": 0.2,
    }

    description = {"idx": "int", "inputs_id": "int"}
    # dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits,
    #                                description=description,
    #                                infinite=False)
    dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits,
                                              description=description,
                                              infinite=False,
                                   batch_size=15,
                                   transform=_map)
    # dataset = TFRecordDataset('../data1.tfrecord', '../data1.index',
    #                           description=description, batch_size=32)
    # dataset = TestDataset()
    # dataset = MyIterableDataset()
    loader = torch.utils.data.DataLoader(dataset, batch_size=8, num_workers=2, prefetch_factor=2)
    for i, data in enumerate(loader):
        # print(i, data['idx'].shape, data['idx'])
        print(i, data['idx'].shape, data['idx'], data['worker_id'])
        # break
        print('='*40)

    # for i, data in enumerate(loader):
    #     print(i, data['idx'])


if __name__ == '__main__':
    # write()
    read()
