import numpy as np

arr = np.array([[0, 1, 2],
                [3, 4, 5],
                [6, 7, 8]])




#高级索引
print(arr[[1,2]]) 

print(arr[...,1]) 
print(arr[:,1]) 

print(arr[:,np.newaxis].shape)