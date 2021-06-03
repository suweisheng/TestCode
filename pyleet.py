#! /usr/bin/env python
# -*- coding: utf-8 -*-

class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def BuildListNode(_list):
    prehead = ListNode(-1)
    prev = prehead
    for num in _list:
        prev.next = ListNode(num)
        prev = prev.next
    return prehead.next

class Solution(object):
    def twoSum(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        ############################## method 1
        n = len(nums)
        for i in xrange(0, n):
            if nums[i+1]:
                for j in xrange(i+1, n):
                    if (nums[i] + nums[j]) == target:
                        return i, j
        ############################## method 2
        hashtable = dict()
        for i, num in enumerate(nums):
            r_num = target - num
            if r_num in hashtable:
                return [hashtable[r_num], i]
            else:
                hashtable[num] = i
# nums = [2,11,7,15]
# target = 9
# ret = Solution().twoSum(nums, target)
# print ret

class Solution(object):
    def reverse(self, x):
        """
        :type x: int
        :rtype: int
        """
        INT_MAX = pow(2, 31) - 1
        INT_MIN = -pow(2, 31)
        rev = 0
        while x != 0:
            # INT_MIN 也是一个负数，不能写成 rev < INT_MIN // 10
            if rev < INT_MIN // 10 + 1 or rev > INT_MAX // 10:
                return 0
            digit = x % 10
            # Python的取模运算在x为负数时也会返回[0, 9)以内的结果，因此这里需要进行特
            # 殊判断
            if x < 0 and digit > 0:
                digit -= 10
            # 同理，Python3 的整数除法在 x 为负数时会向下（更小的负数）取整，因此不能
            # 写成 x //= 10
            x = (x - digit) // 10
            rev = rev * 10 + digit
        return rev
# x = 1463847410
# ret = Solution().reverse(x)
# print ret

class Solution(object):
    def longestCommonPrefix(self, strs):
        """
        :type strs: List[str]
        :rtype: str
        """
        length = len(strs)
        if length == 0:
            return ''
        length, count = len(strs[0]), len(strs)
        ret = ""
        for i in range(length):
            _char = strs[0][i]
            for j in xrange(1, count):
                if i >= len(strs[j]) or strs[j][i] != _char:
                    return strs[0][:i]
        return strs[0]
# strs = ["dogfdssdf","","dogf"]
# ret = Solution().longestCommonPrefix(strs)
# print ret

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        if len(s) % 2 != 0:
            return False
        pairs = {
            ")":"(",
            "}":"{",
            "]":"[",
        }
        stack = list()
        for char in s:
            if char in pairs:
                if not stack or stack[-1] != pairs[char]:
                    return False
                stack.pop()
            else:
                stack.append(char)
        return not stack
# s = "[()]{[][]"
# ret = Solution().isValid(s)
# print ret

class Solution(object):
    def plusOne(self, digits):
        """
        :type digits: List[int]
        :rtype: List[int]
        """
        ############################## method 1
        # if digits[-1] + 1 == 10:
        #     digits[-1] = 0
        # else:
        #     digits[-1] = digits[-1] + 1
        #     return digits
        # for i in xrange(len(digits)-2, -1, -1):
        #     if digits[i] + 1 == 10:
        #         digits[i] = 0
        #     else:
        #         digits[i] = digits[i] + 1
        #         return digits
        # digits.insert(0, 1)
        # return digits
        ############################## method 2
        for i in xrange(len(digits)-1, -1, -1):
            digits[i] = (digits[i] + 1) % 10
            if digits[i] != 0:
                return digits
        digits.insert(0, 1)
        return digits
# digits = [0]
# ret = Solution().plusOne(digits)
# print ret

class Solution(object):
    def mergeTwoLists(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        ############################## method 1
        # 迭代
        # prehead = ListNode(-1)
        # prev = prehead
        # while l1 and l2:
        #     if l1.val <= l2.val:
        #         prev.next= l1
        #         l1 = l1.next
        #     else:
        #         prev.next= l2
        #         l2 = l2.next
        #     prev = prev.next
        # prev.next = l1 if l1 else l2
        # return prehead.next
        ############################## method 2
        # 递归
        if not l1:
            return l2
        elif not l2:
            return l1
        elif l1.val < l2.val:
            l1.next = self.mergeTwoLists(l1.next, l2)
            return l1
        else:
            l2.next = self.mergeTwoLists(l1, l2.next)
            return l2
# l1 = BuildListNode([1,4,8,10,10,15])
# l2 = BuildListNode([1,2,3,4,5,6,7,8,18])
# ret = Solution().mergeTwoLists(l1, l2)
# while ret:
#     print ret.val
#     ret = ret.next
