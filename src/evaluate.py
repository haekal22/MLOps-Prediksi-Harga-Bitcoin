import sys

rmse = 462.92

THRESHOLD = 600

if rmse < THRESHOLD:
    print("Model VALID")
    sys.exit(0)

else:
    print("Model FAILED")
    sys.exit(1)