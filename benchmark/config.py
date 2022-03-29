# TopK = [1, 10, 50, 100, 1000]
# TopK = [1]
# TopK = [1, 2, 4, 8, 16, 32, 64]
# TopK = [1, 2, 4, 8, 16, 32, 64]
TopK = [50]

# NQ = [1, 10, 100, 200, 500, 1000, 1200]
# NQ = [20, 30, 40, 50, 60, 70, 80, 90]
# NQ = [1200]
NQ = [10]
#NQ = [1, 10, 100, 1000, 10000]

# Nprobe = [8, 16, 32, 64, 128, 256, 512]
Nprobe = [16]


NumberOfTestRun = 10000

dim = 128
nb = 1000000
batch = 50000

vectors_per_file = 100000
thread_nums = 10

collection_name = "bench_1"
field_name = "field"
