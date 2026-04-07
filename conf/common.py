import os
import logging

log_level = os.getenv('LOG_LEVEL', 'WARNING').upper()

if log_level == 'DEBUG':
    logging.basicConfig(format="%(filename)s - %(lineno)4d: %(message)s", level=logging.DEBUG)
elif log_level == 'INFO':
    logging.basicConfig(format='%(message)s', level=logging.INFO)
else:
    logging.basicConfig(format='%(message)s', level=logging.WARNING)

DTYPE_FP32 = 4
DTYPE_BF16 = 2
DTYPE_FP16 = 2
DTYPE_FP8 = 1
DTYPE_Q4 = 0.7
DTYPE_INT8 = 1
DTYPE_INT32 = 4
DTYPE_INT64 = 8

# time
US_2_MS = 1e-3
US_2_SEC = 1e-6
MS_2_SEC = 1e-3
MS_2_US = 1e3
SEC_2_US = 1e6

# size
BYTE_2_TB = 1024 ** (-4)
BYTE_2_GB = 1024 ** (-3)
BYTE_2_MB = 1024 ** (-2)
BYTE_2_KB = 1024 ** (-1)
TB_2_BYTE = 1024 ** (4)
GB_2_BYTE = 1024 ** (3)
MB_2_BYTE = 1024 ** (2)
KB_2_BYTE = 1024 ** (1)

# model
MAX_AVG_RATIO = 1.26

# hardware
MEMORY_THRESHOLD_RATIO = 0.9
BLOCK_SIZE = 128
