import sys
import random
import logging
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from reedsolo import RSCodec
from pyldpc import make_ldpc, encode, decode
import math
import cv2 as cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter
# from raptorq import Encoder, Decoder
from enum import Enum

# ********************
# 知识点:
#   线性方程组矩阵解法
#   https://www.shuxuele.com/algebra/systems-linear-equations-matrices.html
#
#   Math is fun
#   https://www.mathsisfun.com/
#
# ********************

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("markcode.wm")
