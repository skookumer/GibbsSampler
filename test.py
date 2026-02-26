from project03 import run_parallel


matrix = run_parallel(k=10, speed="fast", max_iter=10, toprint=True, rtol=1e-30, atol=1e-30, subsample_size=3000000)

ppm = seqlogo.CompletePm(pfm = pfm)
seqlogo.seqlogo(ppm, ic_scale=False, format="png", size="large", filename="test.png")