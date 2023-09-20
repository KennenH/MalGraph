import os
import shutil
import random


def transfer_remote():
    samples_dir = '/root/autodl-tmp'
    all_benign = '/root/autodl-tmp/all_benign'
    one_family_malware = '/root/autodl-tmp/one_family_malware'

    sample = ['malware', 'benign']
    tags = ['test', 'train', 'valid']
    for s in sample:
        index = 0
        for t in tags:
            file_dir = os.path.join(samples_dir, '{}_{}'.format(t, s))
            for file in os.listdir(file_dir):
                dest_dir = all_benign if s == 'benign' else one_family_malware
                shutil.copy(os.path.join(file_dir, file), os.path.join(dest_dir, str(index)))
                index += 1

    delete_all_remote()


def delete_all_remote():
    samples_dir = '/root/autodl-tmp'
    sample = ['malware', 'benign']
    tags = ['test', 'train', 'valid']
    for s in sample:
        for t in tags:
            file_dir = os.path.join(samples_dir, '{}_{}'.format(t, s))
            for f in os.listdir(file_dir):
                os.remove(os.path.join(file_dir, f))


# 重命名pt文件使之与代码相符
def rename(file_dir, mal_or_be, postfix):
    tag_set = ['train', 'test', 'valid']
    for tag in tag_set:
        data_dir = os.path.join(file_dir, '{}_{}{}/'.format(tag, mal_or_be, postfix))
        for index, f in enumerate(os.listdir(data_dir)):
            os.rename(os.path.join(data_dir, f), os.path.join(data_dir, 'm' + f))
    for tag in tag_set:
        data_dir = os.path.join(file_dir, '{}_{}{}/'.format(tag, mal_or_be, postfix))
        for index, f in enumerate(os.listdir(data_dir)):
            os.rename(os.path.join(data_dir, f), os.path.join(data_dir, '{}_{}.pt'.format(mal_or_be, index)))


def split_samples(flag):
    postfix = ''
    file_dir = '/root/autodl-tmp'
    if flag == 'one_family':
        path = os.path.join(file_dir, 'one_family_malware')
        tag = 'malware'
    elif flag == 'standard':
        path = os.path.join(file_dir, 'all')
        postfix = '_backup'
        tag = 'malware'
    elif flag == 'benign':
        path = os.path.join(file_dir, 'all_benign')
        tag = 'benign'
    else:
        print('flag not implemented')
        return

    os_list = os.listdir(path)
    random.shuffle(os_list)
    # 8/1/1 分数据
    train_len = int(len(os_list) * 0.6)
    test_len = int(train_len / 3)
    for index, f in enumerate(os_list):
        if index < train_len:
            shutil.copy(os.path.join(path, f), os.path.join(file_dir, 'train_{}'.format(tag) + postfix))
        elif train_len <= index < train_len + test_len:
            shutil.copy(os.path.join(path, f), os.path.join(file_dir, 'test_{}'.format(tag) + postfix))
        else:
            shutil.copy(os.path.join(path, f), os.path.join(file_dir, 'valid_{}'.format(tag) + postfix))
    rename(file_dir, tag, postfix)


if __name__ == '__main__':
    # transfer_remote()
    delete_all_remote()
    split_samples('one_family')
    split_samples('benign')
