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

class Solution(object):
    def removeDuplicates(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ############################## method 1
        # num = None
        # i = length = 0
        # while i < len(nums):
        #     if num == nums[i]:
        #         nums.pop(i)
        #     else:
        #         num = nums[i]
        #         i += 1
        #         length += 1
        # return length
        ############################## method 2
        # 双指针
        if not nums:
            return 0
        n = len(nums)
        slow = fast = 1
        while fast < n:
            if nums[fast] != nums[fast-1]:
                nums[slow] = nums[fast]
                slow += 1
            fast += 1
        return slow
# nums = [1,1,1,2,3,4,5,5]
# ret = Solution().removeDuplicates(nums)
# print ret
# print nums

class Solution:
    def threeSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort()
        res, k = [], 0
        # 左右指针在两端(k+1, len); k范围[0, len-2]
        for k in range(len(nums) - 2):
            if nums[k] > 0:
                break # because of j > i > k.
            if k > 0 and nums[k] == nums[k - 1]:
                continue # skip the same `nums[k]`.
            i = k + 1
            j = len(nums) - 1
            while i < j:
                s = nums[k] + nums[i] + nums[j]
                if s < 0:
                    i += 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                elif s > 0:
                    j -= 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
                else:
                    res.append([nums[k], nums[i], nums[j]])
                    i += 1
                    j -= 1
                    while i < j and nums[i] == nums[i - 1]:
                        i += 1
                    while i < j and nums[j] == nums[j + 1]:
                        j -= 1
        return res
# nums = [-1,0,1,2,-1,-4]
# nums = []
# nums = [0]
# ret = Solution().threeSum(nums)
# print ret

class Solution(object):
    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        ############################## method 1
        # 计算链表长度
        # def getLength(head):
        #     length = 0
        #     while head:
        #         head = head.next
        #         length += 1
        #     return length
        # # 添加了哑节点，那么头节点的前驱节点就是哑节点本身，此时我们就只需要考虑通用
        # # 的情况即可。
        # dummy = ListNode(-1, head)
        # length = getLength(head)
        # cur = dummy
        # for i in xrange(1, length+1-n):
        #     cur = cur.next
        # cur.next = cur.next.next
        # return dummy.next
        ############################## method 2
        # 栈概念
        # dummy = ListNode(-1, head)
        # stack = list()
        # cur = dummy
        # while cur:
        #     stack.append(cur)
        #     cur = cur.next
        # for i in xrange(n):
        #     stack.pop()
        # prev = stack[-1]
        # prev.next = prev.next.next
        # return dummy.next
        ############################## method 2
        # 双指针
        dummy = ListNode(-1, head)
        second = dummy
        first = head
        for i in xrange(n):
            first = first.next
        while first:
            first = first.next
            second = second.next
        second.next = second.next.next
        return dummy.next
# head = BuildListNode([1,2,3,4])
# n = 2
# ret = Solution().removeNthFromEnd(head, n)
# while ret:
#     print ret.val
    # ret = ret.next

class Solution(object):
    def myPow(self, x, n):
        """
        :type x: float
        :type n: int
        :rtype: float
        """
        ############################## method 1
        # 快速幂
        def quickMul(N):
            if N == 0:
                return 1
            y = quickMul(N // 2)
            return y * y if N % 2 == 0 else y * y * x
        return quickMul(n) if n >= 0 else 1 / quickMul(-n)
        ############################## method 2
        if x == 0: return 0
        res = 1
        if n < 0:
            x = 1 / x
            n = -n
        while n:
            if n & 1:
                res *= x
            x *= x
            n >>= 1
        return res
# x = 2
# n = 5
# ret = Solution().myPow(x, n)
# print ret

def mergeKLists(self, lists):
    """
    :type lists: List[ListNode]
    :rtype: ListNode
    """
    ############################## method 1
    # 顺序合并
    def mergeTwoLists(l1, l2):
        prehead = ListNode(-1)
        prev = prehead
        while l1 and l2:
            if l1.val <= l2.val:
                prev.next= l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 if l1 else l2
        return prehead.next

    # tail = ListNode(-1)
    # for node in lists:
    #     tail = mergeTwoLists(tail, node)
    # return tail.next
    ############################## method 2
    # 分治合并,递归
    # if not lists:
    #     return None
    # def mergeList(lists, l, r):
    #     print l, r
    #     if l == r:
    #         return lists[l]
    #     mid = (l + r) >> 1
    #     one = mergeList(lists, l, mid)
    #     two = mergeList(lists, mid+1, r)
    #     return mergeTwoLists(one, two)
    # return mergeList(lists, 0, len(lists)-1)
    ############################## method 3
    # 迭代
    if not lists:
        return None
    k = len(lists)
    while k > 1:
        idx = 0
        for i in xrange(0, k, 2):
            if i+1 == k:
                lists[idx] = lists[i]
            else:
                lists[idx] = mergeTwoLists(lists[i], lists[i+1])
            idx += 1
        k = idx
    return lists[0]
# a = BuildListNode([1,5,7,10,12])
# b = BuildListNode([1,2,7,10,12])
# c = BuildListNode([1,5,7,10,12])
# lists = [a,b,c]
# ret = Solution().mergeKLists(lists)
# while ret:
#     print ret.val
#     ret = ret.next

class Solution(object):
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        ############################## method 1
        # 双指针
        # nums1_len, nums2_len = len(nums1), len(nums2)
        # if nums1_len + nums2_len == 0:
        #     return None
        # div, mod = divmod((nums1_len+nums2_len), 2)
        # l_p, r_p = 0, 0
        # value1, value2 = None, None
        # for i in xrange(div+1):
        #     value1 = value2
        #     if l_p == nums1_len:
        #         value2 = nums2[r_p]
        #         r_p += 1
        #     elif r_p == nums2_len:
        #         value2 = nums1[l_p]
        #         l_p += 1
        #     else:
        #         if nums1[l_p] < nums2[r_p]:
        #             value2 = nums1[l_p]
        #             l_p += 1
        #         else:
        #             value2 = nums2[r_p]
        #             r_p += 1
        # if mod == 0:
        #     return (value1+value2) / 2.0
        # else:
        #     return float(value2)
        ############################## method 2
        # 分治,二分法,尾递归
        def getKthElement(k):
            index1, index2 = 0, 0
            while True:
                # 特殊情况
                if index1 == m:
                    return nums2[index2 + k - 1]
                if index2 == n:
                    return nums1[index1 + k - 1]
                if k == 1:
                    return min(nums1[index1], nums2[index2])

                # 正常情况
                newIndex1 = min(index1 + k // 2 - 1, m - 1)
                newIndex2 = min(index2 + k // 2 - 1, n - 1)
                pivot1, pivot2 = nums1[newIndex1], nums2[newIndex2]
                if pivot1 <= pivot2:
                    k -= newIndex1 - index1 + 1
                    index1 = newIndex1 + 1
                else:
                    k -= newIndex2 - index2 + 1
                    index2 = newIndex2 + 1

        m, n = len(nums1), len(nums2)
        totalLength = m + n
        if totalLength % 2 == 1:
            return float(getKthElement((totalLength + 1) // 2))
        else:
            left = getKthElement(totalLength // 2)
            right = getKthElement(totalLength // 2 + 1)
            return (left+right) / 2.0

        ############################## method 3
        # if len(nums1) > len(nums2):
        #     nums1, nums2 = nums2, nums1
        # infinty = 2**40
        # m, n = len(nums1), len(nums2)
        # left, right = 0, m
        # # median1：前一部分的最大值
        # # median2：后一部分的最小值
        # median1, median2 = 0, 0

        # while left <= right:
        #     # 前一部分包含 nums1[0 .. i-1] 和 nums2[0 .. j-1]
        #     # // 后一部分包含 nums1[i .. m-1] 和 nums2[j .. n-1]
        #     i = (left + right) // 2
        #     j = (m + n + 1) // 2 - i

        #     # nums_im1, nums_i, nums_jm1, nums_j 分别表示
        #     # nums1[i-1], nums1[i], nums2[j-1], nums2[j]
        #     nums_im1 = (-infinty if i == 0 else nums1[i - 1])
        #     nums_i = (infinty if i == m else nums1[i])
        #     nums_jm1 = (-infinty if j == 0 else nums2[j - 1])
        #     nums_j = (infinty if j == n else nums2[j])

        #     if nums_im1 <= nums_j:
        #         median1 = max(nums_im1, nums_jm1)
        #         median2 = min(nums_i, nums_j)
        #         left = i + 1
        #     else:
        #         right = i - 1
        # if (m + n) % 2 == 0:
        #     return (median1 + median2) / 2.0
        # else:
        #     return float(median1)
# nums1 = [1,2,5,6,7,8]
# nums2 = [3,4]
# ret = Solution().findMedianSortedArrays(nums1, nums2)
# print ret

class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        # 动态规划
        # dp[i]表示i组括号的所有有效组合
        # dp[i] = "(dp[p]的所有有效组合)+【dp[q]的组合】"，其中 1 + p + q = i ,
        # p从0遍历到i-1, q则相应从i-1到0
        dp = [[] for _ in range(n+1)]         # dp[i]存放i组括号的所有有效组合
        print dp
        dp[0] = [""]                          # 初始化dp[0]
        for i in xrange(1, n+1):               # 计算dp[i]
            for p in xrange(i):                # 遍历p
                l1 = dp[p]                    # 得到dp[p]的所有有效组合
                l2 = dp[i-1-p]                # 得到dp[q]的所有有效组合
                for k1 in l1:
                    for k2 in l2:
                        dp[i].append("({0}){1}".format(k1, k2))
        return dp[n]
# n = 3
# ret = Solution().generateParenthesis(n)
# print ret

class Solution(object):
    def divide(self, dividend, divisor):
        """
        :type dividend: int
        :type divisor: int
        :rtype: int
        """
        def div(a, b):
            if a < b: return 0
            count = 1
            tb = b
            while (tb+tb) <= a:
                count += count
                tb += tb
            return count + div(a-tb, b)
        if dividend == 0: return 0
        if divisor == 1: return dividend
        if divisor == -1:
            if dividend == -1<<31:
                return (1<<31) - 1
            else:
                return -dividend
        sign = 1
        if (dividend > 0 and divisor < 0) or (dividend < 0 and divisor > 0):
            sign = -1
        dividend = dividend if dividend > 0 else -dividend
        divisor = divisor if divisor > 0 else -divisor

        res = div(dividend, divisor)
        if sign == 1:
            return res
        else:
            return -res
# dividend = 7
# divisor = -3
# ret = Solution().divide(dividend, divisor)
# print ret

class Solution(object):
    def isValidSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: bool
        """
        # init data
        rows = [{} for i in xrange(9)]
        columns = [{} for i in xrange(9)]
        boxes = [{} for i in xrange(9)]
        for i in xrange(9):
            for j in xrange(9):
                num = board[i][j]
                if num == '.': continue
                box_index = (i // 3) * 3 + j // 3
                # keep the current cell value
                rows[i][num] = rows[i].get(num, 0) + 1
                columns[j][num] = columns[j].get(num, 0) + 1
                boxes[box_index][num] = boxes[box_index].get(num, 0) + 1
                # check if this value has been already seen before
                if (rows[i][num] > 1
                    or columns[j][num] > 1
                    or boxes[box_index][num] > 1):
                    return False
        return True
# board = [
#     ["5","5",".",".","7",".",".",".","."],
#     ["6",".",".","1","9","5",".",".","."],
#     [".","9","8",".",".",".",".","6","."],
#     ["8",".",".",".","6",".",".",".","3"],
#     ["4",".",".","8",".","3",".",".","1"],
#     ["7",".",".",".","2",".",".",".","6"],
#     [".","6",".",".",".",".","2","8","."],
#     [".",".",".","4","1","9",".",".","5"],
#     [".",".",".",".","8",".",".","7","9"],
# ]
# ret = Solution().isValidSudoku(board)
# print ret