import pandas as pd
import numpy as np
frame = pd.DataFrame(np.random.randn(4, 3), columns=list('bde'), index=['Utah', 'Ohio', 'Texas', 'Oregon'])
f = lambda x: x.max() - x.min()
frame = frame.apply(f)
print(frame)

cal = A.sum(axis=0)
cal.reshape(1,4)