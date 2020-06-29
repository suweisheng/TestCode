#! /usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import time

# if sys.platform == 'win32':
#     reload(sys)
#     sys.setdefaultencoding('gbk')
'''
    range, xrange 在 python2.7 都存在, , xrange是生成一个迭代器, 在 python3 只存在xrange, 但实现是range的实现
    range 会直接生成一个list对象
    range 是生成一个序列,而xrange则不会直接生成一个list，而是每次调用返回其中的一个值
    range(2, 4, 2) [a,b)

    enumerate(seq, start) 生成一个序列迭代器,带key-val的元表
'''

def loop_list():
    raw_list = ['html', 'js', 'css', 'python']
    # method 1
    for i in raw_list:
        print "index:%d, value:%s" % (raw_list.index(i), i)
    # method 2
    for i in xrange(len(raw_list)):
        print "index:%d, value:%s" % (i, raw_list[i])
    # method 3
    for i, val in enumerate(raw_list):
        print "index:%d, value:%s" % (i, val)
    # method 3
    for i, val in enumerate(raw_list, 2):
        print "index:%d, value:%s" % (i-2, val)
    """"""

    raw_dict = {"a":"a1", "b":"b2"}
    # method 1
    for k in raw_dict.keys():
        print "key:%s, value:%s" % (k, raw_dict[k])
    # method 2
    key_list = list(filter(lambda k: raw_dict.get(k) == "88", raw_dict.keys()))
    for value in raw_dict.values():
        print "key:%s, value:%s" % (k, "s")
    # method 3
    for keys, v in raw_dict.items():
        print "key:%s, value:%s" % (k, v)

def count_running_time(cb_func):
    begin_time = time.clock()
    cb_func()
    end_time = time.clock()
    return u"耗时：%s" % str(end_time - begin_time)


def singleNumber():
    '''
    给定一个非空整数数组，除了某个元素只出现一次以外，其余每个元素均出现两次。找出那个只出现了一次的元素。
    你的算法应该具有线性时间复杂度。 你可以不使用额外空间来实现吗？
    '''
    array = [5, 3, 4, 5, 6, 4, 2, 3]
    # ==>>> mine
    # value = None
    # for i in xrange(len(array)):
    #     if array[i] == "x":
    #         continue
    #     for j in xrange(i+1, len(array)):
    #         if array[i] == array[j]:
    #             array[i], array[j] = "x", "x"
    #             break
    #     if array[i] != "x":
    #         value = array[i]
    #         break
    # print value

    # ==>>> other
    # 异或交换律和结合律
    # 交换律：a ^ b ^ c <=> a ^ c ^ b
    # 任何数于0异或为任何数 0 ^ n => n
    # 相同的数异或为0: n ^ n => 0
    # 时间复杂度O(N) 空间复杂度O(1)
    value = 0
    for x in array:
        value = value ^ x
    print value

def majorityElement():
    '''
    给定一个大小为 n 的数组，找到其中的多数元素。多数元素是指在数组中出现次数大于 ⌊ n/2 ⌋ 的元素。
    你可以假设数组是非空的，并且给定的数组总是存在多数元素。
    '''
    array = [2, 2, 1, 1, 1, 2, 2]
    # ==>>> mine
    # count_dict = {}
    # for x in array:
    #     if count_dict.has_key(x):
    #         count_dict[x] = count_dict[x] + 1
    #     else:
    #         count_dict[x] = 0
    # value = None
    # for k, v in count_dict.items():
    #     if v > (len(count_dict)//2):
    #         value = v if (not value or v > value) else value
    # print value

    # ==>>> mine
    # array.sort()
    # print array[len(array)/2]

    # ==>>> other
    # 摩尔投票法：
    # 核心就是对拼消耗。
    # 玩一个诸侯争霸的游戏，假设你方人口超过总人口一半以上，并且能保证每个人口出去干仗都能一对一同归于尽。最后还有人活下来的国家就是胜利。
    # 那就大混战呗，最差所有人都联合起来对付你（对应你每次选择作为计数器的数都是众数），或者其他国家也会相互攻击（会选择其他数作为计数器的数），但是只要你们不要内斗，最后肯定你赢。
    # 最后能剩下的必定是自己人。
    # 时间复杂度O(N) 空间复杂度O(1)
    value, count = 0, 0
    for v in array:
        if count == 0:
            value = v
            count += 1
        else:
            if v == value:
                count += 1
            else:
                count -= 1
    print value

def searchMatrix():
    '''
    编写一个高效的算法来搜索 m x n 矩阵 matrix 中的一个目标值 target。该矩阵具有以下特性：
    每行的元素从左到右升序排列。
    每列的元素从上到下升序排列。
    '''
    array = [
        [1, 4, 7, 11, 15],
        [2, 5, 8, 12, 19],
        [3, 6, 9, 16, 22],
        [10, 13, 14, 17, 24],
        [18, 21, 23, 26, 30],
    ]



if __name__ == "__main__":
    # print count_running_time(singleNumber)
    # print count_running_time(majorityElement)
    print count_running_time(searchMatrix)