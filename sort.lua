local M = {}

------------------------------------------------ some function tools


------------------------------------------------ some sort function
function M.bubble(list)
    for i=1, #list-1 do
        local is_swap = false
        for j=1, #list-i do
            if list[j+1] < list[j] then
                list[j], list[j+1] = list[j+1], list[j]
                is_swap = is_swap or true
            end
        end
        if not is_swap then break end
    end
    return list
end

function M.choise(list)
    for i=1, #list-1 do
        local min_index = i
        for j=i+1, #list do
            if list[min_index] > list[j] then
                min_index = j
            end
        end
        list[i], list[min_index] = list[min_index], list[i]
    end
    return list
end

function M.insert(list)
    for i=2, #list do
        local min_value = list[i]
        for j=i, 2, -1 do
            if min_value < list[j-1] then
                list[j] = list[j-1]
                i = j - 1
            end
        end
        list[i] = min_value
    end
    return list
end

function M.quick(list)
    local function __quick(_list, left, right)
        if left >= right then return end
        local base = _list[left]
        local i, j = left, right
        while i < j do
            -- j must be forward begor i
            while i < j and _list[j] > base do
                j = j - 1
            end
            while i < j and _list[i] <= base do
                i = i + 1
            end
            if i < j then
                _list[i], _list[j] = _list[j], _list[i]
            end
        end
        _list[i], _list[left] = _list[left], _list[i]
        __quick(_list, left, i-1)
        __quick(_list, i+1, right)
    end
    __quick(list, 1, #list)
    return list
end

function M.merge(list)
    local function __merge(left, mid, right)
        local i, j  = left, mid+1
        local tmp, k = {}, 1
        while i <= mid and j <= right do
            if list[i] < list[j] then
                tmp[k] = list[i]
                i = i + 1
            else
                tmp[k] = list[j]
                j = j + 1
            end
            k = k + 1
        end
        while i <= mid do
            tmp[k] = list[i] 
            k = k + 1
            i = i + 1
        end
        while j <= right do
            tmp[k] = list[j] 
            k = k + 1
            j = j + 1
        end
        for i=1, #tmp do
            list[left+i-1] = tmp[i]
        end
    end
    local function __separate(left, right)
        if left >= right then return end
        local mid =  (left+right) // 2
        __separate(left, mid)
        __separate(mid+1, right)
        __merge(left, mid, right)
    end
    __separate(1, #list)
    return list
end

function M.shell(list)
    local gap = #list // 2
    while gap >= 1 do
        for i=gap+1, #list do
            local min_value = list[i]
            for j=i, gap+1,-gap do
                if min_value < list[j-gap] then
                    list[j] = list[j-gap]
                    i = j - gap
                end
            end
            list[i] = min_value
        end
        gap = gap // 2
    end
    return list
end

function M.heap(list)
    local __adjust_down = function(root, length)
        local child = root * 2
        local value = list[root]
        while child <= length do
            if child < length and list[child] < list[child+1] then
                child = child + 1
            end
            if value < list[child] then
                list[child//2] = list[child]
                child = child * 2
            else
                break
            end
        end
        list[child//2] = value
    end
    for i=(#list//2), 1, -1 do
        __adjust_down(i, #list)
    end
    for i=#list, 2, -1 do
        list[1], list[i] = list[i], list[1]
        __adjust_down(1, i-1)
    end
    return list
end

function M.count(list)
    -- list attr muse be integer
    local v_min , v_max = nil, nil
    for _, i in ipairs(list) do
        if not v_min or i < v_min then
            v_min = i
        end
        if not v_max or i > v_max then
            v_max = i
        end
    end
    local count_list = {}
    for i=1, v_max-v_min+1 do
        count_list[i] = 0
    end
    for _, i in ipairs(list) do
        -- i might be negative -123
        local index = i - v_min + 1
        count_list[index] = count_list[index] + 1
    end
    for i=2, #count_list do
        -- accumulation, so can sort same element
        count_list[i] = count_list[i] + count_list[i-1]
    end
    local new_list = {}
    for i=#list, 1, -1 do
        local index = list[i] - v_min + 1
        new_list[count_list[index]] = list[i]
        count_list[index] = count_list[index] - 1
    end
    for i, j in ipairs(new_list) do
        list[i] = j
    end
    return list
end

function M.binary(list)
    local left, mid, right
    for i=2, #list do 
        local value = list[i]
        local left, right = 1, i-1
        while left <= right do
            mid = (left+right) // 2
            if value < list[mid] then
                right = mid - 1
            else
                left = mid + 1
            end
        end
        for j=i, left+1, -1 do
            list[j] = list[j-1]
        end
        list[left] = value
    end
    return list
end

------------------------------------------------ test

local function print_list(list)
    local str = "["
    for k, v in ipairs(list) do
        str = str..v..", "
    end
    str = str.."]"
    print(str)
end

local function copy(obj)
    local copy_tb = {}
    for k, v in pairs(obj) do
        copy_tb[k] = v
    end
    return copy_tb
end

local raw_math_random = math.random
math.random = function(a, b)
    math.randomseed(os.time())
    return raw_math_random(a, b)
end

local function shuffle(tb)
    local j
    for i=#tb, 2, -1 do
        j = math.random(1, i)
        tb[i], tb[j] = tb[j], tb[i]
    end
    return tb
end

local function test()
    local test_list = {-3, 0, 2, 2, 9, 4, 3, 8, 1, 7, -4, -8, 6, 5, }
    print_list(M.bubble(copy(test_list)))
    print_list(M.choise(copy(test_list)))
    print_list(M.insert(copy(test_list)))
    print_list(M.quick(copy(test_list)))
    print_list(M.merge(copy(test_list)))
    print_list(M.shell(copy(test_list)))
    print_list(M.heap(copy(test_list)))
    print_list(M.binary(copy(test_list)))
    print("test success !")
end

test()

return M