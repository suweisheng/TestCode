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

    def strStr(self, haystack, needle):
        """
        :type haystack: str
        :type needle: str
        :rtype: int
        """
        ############################## method 1
        # 暴力法
        n_len = len(needle)
        if n_len == 0:
            return 0
        h_len = len(haystack)
        for i in xrange(h_len-n_len+1):
            flag = True
            for j in xrange(n_len):
                if haystack[i+j] != needle[j]:
                    flag = False
                    break
            if flag:
                return i
        return -1
        ############################## method 2
        # 双指针法
        n_len = len(needle)
        h_len = len(haystack)
        for i in xrange(h_len-n_len+1):
            a = i; b = 0
            while b < n_len and haystack[a] == needle[b]:
                a += 1
                b += 1
            if b == (n_len-1):
                return i
        return -1
# haystack = "fduhf"
# needle = "hf"
# ret = Solution().strStr(haystack, needle)
# print ret

class Solution(object):
    def mySqrt(self, x):
        """
        :type x: int
        :rtype: int
        """
        ############################## method 1
        # 二分查找
        left, right = 0, x
        ans = -1
        while left <= right:
            mid = (left + right) // 2
            if mid * mid <= x:
                left = mid + 1
                ans = mid
            else:
                right = mid - 1
        return ans
# x = 10
# ret = Solution().mySqrt(x)
# print ret

class Solution(object):
    def defangIPaddr(self, address):
        """
        :type address: str
        :rtype: str
        """
        new_str = ""
        for x in address:
            if x == ".":
                new_str = new_str + "[.]"
            else:
                new_str = new_str + x
        return new_str
# address = "1.1.1.1"
# ret = Solution().defangIPaddr(address)
# print ret

class Solution(object):
    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        head = curr = ListNode(-1)
        carry = val = 0
        while carry or l1 or l2:
            val = carry
            if l1:
                l1, val = l1.next, l1.val + val
            if l2:
                l2, val = l2.next, l2.val + val
            carry, val = divmod(val, 10)
            curr.next = ListNode(val)
            curr = curr.next
            # curr.next = curr = ListNode(val)
        return head.next
# l1 = BuildListNode([1,4,8,5,3,2])
# l2 = BuildListNode([1,2,3,4,5,6,1,0,5])
# ret = Solution().addTwoNumbers(l1, l2)
# while ret:
#     print ret.val
#     ret = ret.next

class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        ############################## method 1
        # length = len(s)
        # l_max = 0
        # for i in xrange(length):
        #     mapp = set(s[i])
        #     temp = 1
        #     for j in xrange(i+1, length):
        #         if s[j] in mapp:
        #             break
        #         else:
        #             temp = temp + 1
        #             mapp.add(s[j])
        #     if temp > l_max:
        #         l_max = temp
        # return l_max
        ############################## method 2
        l, r = 0, 0
        length = 0
        mapp = set()
        while r < len(s):
            if s[r] in mapp:
                # 左指针移动
                mapp.remove(s[l])
                l += 1
            else:
                # 右指针移动
                mapp.add(s[r])
                r += 1
            # 每次记录最大长度
            if r - l > length:
                length = r - l
            # length = max(r-i, length)
        return length
# s = "abcabcbb"
# ret = Solution().lengthOfLongestSubstring(s)
# print ret
