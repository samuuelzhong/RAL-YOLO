from bigO import BigO

def your_algorithm(data):
    result = 0
    for i in range(len(data)):
        for j in range(len(data)):
            result += data[i] * data[j]
    return result

# 分析时间复杂度
lib = BigO()
complexity = lib.test(your_algorithm, "random")
print(complexity)