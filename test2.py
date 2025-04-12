# import pandas as pd
# from collections import Counter
# from matplotlib import pyplot as plt
# ls = [1,2,2,2,3,4,4,4,5,5,6,5,3,1,8,8,9,9,0]
# b = sorted(Counter(ls).items())
# b_dict = dict(b)
# print(b_dict)
#
# plt.bar(b_dict.keys(),b_dict.values())
#
# plt.show()

import torch
print(torch.__version__)
print(torch.cuda.device_count())

