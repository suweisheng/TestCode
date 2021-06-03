
local raw_print = print
local print = function(obj)
    if type(obj) == "table" then
        local keys = {}
        for i, j in ipairs(obj) do
            table.insert(keys, i)
        end
        table.sort(keys)
        local is_list = keys[1] == 1 and keys[#keys] == #keys
        local str = ""
        if is_list then
            for _, v in ipairs(obj) do
                str = str..v..", "
            end
        else
            for k, v in pairs(obj) do
                str = str .. "["..k.."]"..":"..v..", "
            end
        end
        str = "{"..string.sub(str, 1, -3).."}"
        raw_print(str)
    else
        raw_print(obj)
    end
end

function ListNode(val, next)
    return {val=val, next=next}
end

function BuildListNode(_list)
    local prehead = ListNode(-1)
    local prev = prehead
    for _, num in ipairs(_list) do
        prev.next = ListNode(num)
        prev = prev.next
    end
    return prehead.next
end

function twoSum(nums, target)
    --[[
    :type nums: list[int]
    :type target: number
    :rtype: list[int]
    ]]--
    ------------------------------ method 1
    local n = #nums
    for i=1, n do
        for j=i+1, n do
            if (nums[i] + nums[j]) == target then
                return {i, j}
            end
        end
    end
    ------------------------------ method 2
    local hashtable = {}
    for i, num in ipairs(nums) do
        local r_num = target - num
        if hashtable[r_num] then
            return {hashtable[r_num], i}
        else
            hashtable[num] = i
        end
    end
end
-- local nums = {1,11,2,7}
-- local target = 9
-- local ret = twoSum(nums, target)
-- print(ret)

function reverse(x)
    --[[
    :type x: 32int
    :rtype: 32int
    ]]--
    local INT_MIN = -2<<30
    local INT_MAX = (2<<30) - 1
    local rev = 0
    while x ~= 0 do
        if rev < (INT_MIN//10+1) or rev > (INT_MAX//10) then
            return 0
        end
        local digit = x % 10
        if x < 0 and digit > 0 then
            digit = digit - 10
        end
        x = (x - digit) // 10
        rev = rev * 10 + digit
    end
    return rev
end
-- local x = -21554
-- local ret = reverse(x)
-- print(ret)

function longestCommonPrefix(strs)
    --[[
    :type strs: string
    :rtype: string
    ]]--
    if not strs[1] then
        return ""
    end
    local length, count = string.len(strs[1]), #strs
    for i=1, length do
        local _char = string.sub(strs[1], i, i)
        for j=2, count do
            if i > string.len(strs[j])
                or string.sub(strs[j], i, i) ~=_char
                then
                return string.sub(strs[1], 1, i-1)
            end
        end
    end
    return strs[1]
end
-- local strs = {"dogfdssdf","dogfdssdf","dogf"}
-- local ret = longestCommonPrefix(strs)
-- print(ret)

function isValid(s)
   --[[
    :type s: string
    :rtype: boolean
    ]]--
    if string.len(s) % 2 ~= 0 then
        return false
    end
    local pairs = {
        [")"] = "(",
        ["}"] = "{",
        ["]"] = "[",
    }
    local stack = {}
    for i = 1, string.len(s) do
        local char = string.sub(s, i, i)
        if pairs[char] then
            local _c = table.remove(stack)
            if not _c or pairs[char] ~= _c then
                return false
            end
        else
            table.insert(stack, char)
        end
    end
    if next(stack) then
        return false
    end
    return true
end
-- local s = "[([]{})][][]"
-- local ret = isValid(s)
-- print(ret)

function plusOne(digits)
    --[[
    :type digits: list[int]
    :rtype: list[int]
    ]]--
    ------------------------------ method 1
    -- if digits[#digits] + 1 == 10 then
    --     digits[#digits] = 0
    -- else
    --     digits[#digits] = digits[#digits] + 1
    --     return digits
    -- end
    -- for i = #digits-1, 1, -1 do
    --     if digits[i] + 1 == 10 then
    --         digits[i] = 0
    --     else
    --         digits[i] = digits[i] + 1
    --         return digits
    --     end
    -- end
    -- table.insert(digits, 1, 1)
    -- return digits
    ------------------------------ method 2
    for i = #digits, 1, -1 do
        digits[i] = (digits[i] + 1) % 10
        if digits[i] ~= 0 then
            return digits
        end
    end
    table.insert(digits, 1, 1)
    return digits
end
-- local digits = {9,9,9}
-- local ret = plusOne(digits)
-- print(ret)

function mergeTwoLists(l1, l2)
    --[[-
    :type l1: ListNode
    :type l2: ListNode
    :rtype: ListNode
    -]]
    ------------------------------ method 1
    -- 迭代
    -- local prehead = ListNode(-1)
    -- local prev = prehead
    -- while l1 and l2 do
    --     if l1.val <= l2.val then
    --         prev.next = l1
    --         l1 = l1.next
    --     else
    --         prev.next = l2
    --         l2 = l2.next
    --     end
    --     prev = prev.next
    -- end
    -- if l1 then
    --     prev.next = l1
    -- else
    --     prev.next = l2
    -- end
    -- return prehead.next
    ------------------------------ method 2
    -- 递归
    if not l1 then
        return l2
    elseif not l2 then
        return l1
    elseif l1.val <= l2.val then
        l1.next = mergeTwoLists(l1.next, l2)
        return l1
    else
        l2.next = mergeTwoLists(l1, l2.next)
        return l2
    end
end
-- local l1 = BuildListNode({1,4,8,10,10,15})
-- local l2 = BuildListNode({1,2,3,4,5,6,7,8,18})
-- local ret = mergeTwoLists(l1, l2)
-- while ret do
--     print(ret.val)
--     ret = ret.next
-- end