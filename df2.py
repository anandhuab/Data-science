# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 12:44:58 2022

@author: DELL
"""

import pandas as pd
st={'unit test-1':[5,6,7,8],'unit test-2':[1,2,1,2]}
st2={'unit test-1':[1,2,3,4],'unit test-2':[1,2,1,2]}

ds=pd.DataFrame(st)
ds2=pd.DataFrame(st2)
print(ds)
print(ds2)
print("subtraction")
print(ds.sub(ds2))