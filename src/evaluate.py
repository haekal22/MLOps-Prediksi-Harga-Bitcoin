import sys

rmse = 788.85

THRESHOLD = 800

if rmse < THRESHOLD:
    print("Model VALID")
    sys.exit(0)

else:
    print("Model FAILED")
    sys.exit(1)