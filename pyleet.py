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

class Solution(object):
    def myAtoi(self, s):
        """
        :type s: str
        :rtype: int
        """
        state = "start"
        sign = 1
        ans = 0
        # 状态机
        tb = {
            "start" : ["start", "signed", "in_number", "end"],
            "signed" : ["end", "end", "in_number", "end"],
            "in_number" : ["end", "end", "in_number", "end"],
            "end" : ["end", "end", "end", "end"],
        }
        INT_MAX = (1 << 31) - 1
        INT_MIN = - (1 << 31)
        for c in s:
            index = None
            if c.isspace():
                index = 0
            elif c == "+" or c == "-":
                index = 1
            elif c.isdigit():
                index = 2
            else:
                index = 3
            state = tb[state][index]
            if state == "in_number":
                if (ans > (INT_MAX//10)
                    or (ans == (INT_MAX//10) and int(c) > 7)):
                    if sign == 1:
                        return INT_MAX
                    else:
                        return INT_MIN
                ans = ans * 10 + int(c)
            elif state == "signed":
                sign = 1 if c == "+" else -1
            elif state == "end":
                break
        return sign * ans
# s = "2147483648"
# ret = Solution().myAtoi(s)
# print ret

class Solution(object):
    def permute(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        ############################## method 1
        # 动态规划
        # length = len(nums)
        # dp = [[] for x in xrange(length)]
        # dp[0] = [nums[0:1]]
        # for i in xrange(1, length):
        #     insert_list = [nums[i]]
        #     for _list in dp[i-1]:
        #         _len = len(_list)
        #         for j in xrange(_len+1):
        #             temp = _list[0:j] + insert_list + _list[j:_len]
        #             dp[i].append(temp)
        # return dp[length-1]
        ############################## method 2
        # 回溯
        def backtrack(first = 0):
            # 所有数都填完了
            if first == n:  
                res.append(nums[:])
            for i in range(first, n):
                # 动态维护数组
                nums[first], nums[i] = nums[i], nums[first]
                # 继续递归填下一个数
                backtrack(first + 1)
                # 撤销操作
                nums[first], nums[i] = nums[i], nums[first]
        n = len(nums)
        res = []
        backtrack()
        return res
# nums = [1,2,3,4,5,6]
# ret = Solution().permute(nums)
# print len(ret)

class Solution(object):
    def searchRange(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        ############################## method 1
        # l, r = 0, len(nums)-1
        # index = None
        # while l <= r:
        #     mid = l + (r - l) // 2
        #     if nums[mid] < target:
        #         l = mid + 1
        #     elif nums[mid] > target:
        #         r = mid - 1
        #     else:
        #         index = mid
        #         break
        # if index == None:
        #     return [-1, -1]
        # start = end = index
        # while start >= 1:
        #     if nums[start-1] == target:
        #         start -= 1
        #     else:
        #         break
        # while (end+1) < len(nums):
        #     if nums[end+1] == target:
        #         end += 1
        #     else:
        #         break
        # return [start, end]
        ############################## method 2
        def binarySearch(nums, target, lower):
            length = len(nums)
            left, right = 0, length-1
            ans = length
            while left <= right:
                mid = left + (right-left) / 2
                if target < nums[mid] or (lower and target <= nums[mid]):
                    right = mid - 1
                    ans = mid
                else:
                    left = mid + 1
            return ans
        # 第一个 >= target 的数
        leftIdx = binarySearch(nums, target, True)
        if leftIdx >= len(nums) or nums[leftIdx] != target:
            return [-1, -1]
        # 第一个 > target 的数 -1
        rightIdx = binarySearch(nums, target, False) - 1
        return [leftIdx, rightIdx]
# nums = [1,1,1,1,1,2]
# target = 32
# ret = Solution().searchRange(nums, target)
# print ret

class Solution(object):
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        left, right = 0, len(nums)-1
        while left <= right:
            mid = (left+right) // 2
            if nums[mid] == target:
                return mid
            # 注意要<=,不能<,特殊：left=1, right=2, mid=1
            if nums[left] <= nums[mid]:
                # 有序
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                # 无序
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return -1
# nums = [2,1]
# target = 1
# ret = Solution().search(nums, target)
# print ret

class Solution(object):
    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        n = len(s)
        if n < 2:
            return s
        dp = [[False]*n for _ in xrange(n)]
        res = s[0]
        for i in xrange(n):
            dp[i][i] = True
        # 子串长度
        for L in xrange(2, n+1):
            # 枚举左边界，左边界的上限设置可以宽松一些
            # 如果右边界越界，就可以退出当前循环
            for i in xrange(n-L+1):
                # 由 L 和 i 可以确定右边界，即 j - i + 1 = L 得
                j = L + i - 1
                if s[i] == s[j]:
                    if j - i < 3:
                        dp[i][j] = True
                    else:
                        dp[i][j] = dp[i+1][j-1]
                else:
                    dp[i][j] = False
                if dp[i][j] and L > len(res):
                    res = s[i:j+1]
        return res
# s = "bdb"
# ret = Solution().longestPalindrome(s)
# print ret

class Solution(object):
    def maxArea(self, height):
        """
        :type height: List[int]
        :rtype: int
        """
        l, r = 0, len(height) - 1
        ans = 0
        while l < r:
            area = min(height[l], height[r]) * (r - l)
            ans = max(ans, area)
            if height[l] <= height[r]:
                l += 1
            else:
                r -= 1
        return ans
# height = [1,8,6,2,5,4,8,3,7]
# ret = Solution().maxArea(height)
# print ret

class Solution(object):
    def findDuplicate(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        ############################## method 1
        # 二分查找
        # n = len(nums)
        # l, r = 0, n-1
        # ans = -1
        # while l <= r:
        #     mid = (l + r) // 2
        #     cnt = 0
        #     for x in nums:
        #         if x <= mid:
        #             cnt += 1
        #     if cnt <= mid:
        #         l = mid + 1
        #     else:
        #         r = mid - 1
        #         ans = mid
        # return ans
        ############################## method 2
        # 快慢指针
        slow, fast = 0, 0
        while True:
            slow = nums[slow]
            fast = nums[nums[fast]]
            if slow == fast: break
        # 寻找起始点
        slow = 0
        while True:
            slow = nums[slow]
            fast = nums[fast]
            if slow == fast: break
        return slow
# nums = [4,3,2,1,4]
# ret = Solution().findDuplicate(nums)
# print ret

class Solution(object):
    def letterCombinations(self, digits):
        """
        :type digits: str
        :rtype: List[str]
        """
        ############################## method 1
        # 深度优先搜索、递归的思想
        if not digits:
            return []
        phoneMap = {
            "2": "abc",
            "3": "def",
            "4": "ghi",
            "5": "jkl",
            "6": "mno",
            "7": "pqrs",
            "8": "tuv",
            "9": "wxyz",
        }
        n = len(digits)
        if n < 1:
            return list()
        def backtrack(index):
            if index == n:
                combinations.append("".join(combination))
            else:
                digit = digits[index]
                for letter in phoneMap[digit]:
                    combination.append(letter)
                    backtrack(index + 1)
                    combination.pop()
        combinations = list()
        combination = list()
        backtrack(0)
        return combinations
# digits = "23"
# ret = Solution().letterCombinations(digits)
# print ret

class Solution(object):
    def subsets(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        length = len(nums)
        def dfs(index):
            if index == length:
                ans.append(list(path))
                return
            path.append(nums[index])
            dfs(index+1)
            path.pop()
            dfs(index+1)
        ans = list()
        path = list()
        dfs(0)
        return ans
# nums = [1,2,3]
# ret = Solution().subsets(nums)
# print ret

class Solution(object):
    def countAndSay(self, n):
        """
        :type n: int
        :rtype: str
        """
        def pingjiestr(_str):
            num, count = _str[0], 1
            ans = ""
            for i in xrange(1, len(_str)):
                if num == _str[i]:
                    count += 1
                else:
                    ans += str(count) + str(num)
                    num = _str[i]
                    count = 1
            ans += str(count) + str(num)
            return ans

        pre_ans = "1"
        for i in xrange(n-1):
            pre_ans = pingjiestr(pre_ans)
        return pre_ans
# n = 30
# ret = Solution().countAndSay(n)
# print ret

class Solution(object):
    def rotate(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: None Do not return anything, modify matrix in-place instead.
        """
        ############################## method 1
        # 暴力解，要额外开辟等额的空间
        # n = len(matrix)
        # matrix_new = [[0] * n for _ in range(n)]
        # for i in range(n):
        #     for j in range(n):
        #         matrix_new[j][n - i - 1] = matrix[i][j]
        # # 不能写成 matrix = matrix_new
        # matrix[:] = matrix_new
        ############################## method 2
        n = len(matrix)
        for i in xrange(n//2):
            for j in xrange((n+1)//2):
                print i, j
                temp = matrix[i][j];
                matrix[i][j] = matrix[n-j-1][i]
                matrix[n-j-1][i] = matrix[n-i-1][n-j-1]
                matrix[n-i-1][n-j-1] = matrix[j][n-i-1]
                matrix[j][n-i-1] = temp
# matrix = [[1,2,3,4],[5,6,7,8],[9,10,11,12],[13,14,15,16]]
# ret = Solution().rotate(matrix)
# print matrix

class Solution(object):
    def groupAnagrams(self, strs):
        """
        :type strs: List[str]
        :rtype: List[List[str]]
        """
        import collections
        ret = collections.defaultdict(list)
        for _str in strs:
            key = "".join(sorted(_str))
            ret[key].append(_str)
        return ret.values()
# strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
# ret = Solution().groupAnagrams(strs)
# print ret

class Solution(object):
    def uniquePaths(self, m, n):
        """
        :type m: int
        :type n: int
        :rtype: int
        """
        # f = [[1] * n] + [[1] + [0] * (n - 1) for _ in range(m - 1)]
        f = [[0 for _ in xrange(n)] for _ in xrange(m)]
        for i in xrange(m):
            f[i][0] = 1
        for i in xrange(n):
            f[0][i] = 1
        for i in xrange(1, m):
            for j in xrange(1, n):
                f[i][j] = f[i-1][j] + f[i][j-1]
        return f[m-1][m-1]
# m = 4
# n = 5
# ret = Solution().uniquePaths(m, n)
# print ret