import os

os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')  # torch and lightgbm each ship OpenMP on Windows; import kit first
