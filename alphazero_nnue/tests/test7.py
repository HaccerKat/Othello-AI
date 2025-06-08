import time
# append
n = 10**6
start = time.perf_counter()
lst = []
for i in range(n):
    lst.append(i)
end = time.perf_counter()
print("Time for append:", end - start)
start = time.perf_counter()
# preallocate
lst = [0] * n
for i in range(n):
    lst[i] = i
end = time.perf_counter()
print("Time for index:", end - start)
# index is actually slower
