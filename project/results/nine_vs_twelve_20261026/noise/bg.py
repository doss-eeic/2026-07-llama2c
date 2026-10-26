import sys, time
busy = float(sys.argv[1]) * 0.010
while True:
    t = time.perf_counter()
    while time.perf_counter() - t < busy:
        pass
    time.sleep(0.010 - busy)
