import numpy as np

import matplotlib.pyplot as plt




#random numbers from 0 to 1 (uniform distribution)
R100 = np.random.rand(100, 100)
print(R100)



#get U,S,V^T from SVD for R100
U, S, VT = np.linalg.svd(R100,full_matrices=False)
S = np.diag(S)




#Singular Values plot of R100
plt.figure(1)
plt.semilogy(np.diag(S))
plt.title('Singular Values of R100')
plt.show()





#Create one hundred 100x100 matrices
lst = [] 
for i in range(100):
    lst.append(np.random.rand(100, 100)) 

#get SVD for the 100 matrices
RU=[]
RS=[]
RVT=[]
for i in range(100):
  U, S, VT = np.linalg.svd(lst[i],full_matrices=False)
  S = np.diag(S)
  RU.append(U)
  RS.append(S)
  RVT.append(VT)
  

#Singular Values plot of the 100 matrices
for i in range(100):
 plt.figure(i)
 plt.semilogy(np.diag(RS[i]))
 plt.title('Singular Values of Random Matrix {}'.format(i+1))
 plt.show()
 





#box and whisker plot for the SV distribution for the first 4 of our matrices
data=[np.diag(RS[0]),np.diag(RS[1]),np.diag(RS[2]),np.diag(RS[3])]

fig = plt.figure(figsize =(10, 7))
# Creating axes instance
ax = fig.add_axes([0, 0, 1, 1])
plt.ylim([0, 10]) 
# Creating plot
bp = ax.boxplot(data)
plt.show()







#define mean and median of singular values functions
def Mean(r):
   return (np.sum(np.diag(RS[r])))/100


def Median(r):
   return 0.5*(np.diag(RS[r])[49]+np.diag(RS[r])[50])



#plots for the mean and median functions
x = [i for i in range(0, 100)]

y = [Mean(x[i]) for i in range(0,100)]


plt.figure()
plt.plot(x,y, color ='tab:blue')
plt.title('Mean of Singular Values')
plt.xlim([0, 99]) 
plt.ylim([2.25, 3.5]) 
plt.show()



y = [Median(x[i]) for i in range(0,100)]

plt.figure()
plt.plot(x,y, color ='tab:blue')
plt.title('Median of Singular Values')
plt.xlim([0, 99]) 
plt.ylim([2, 4]) 
plt.show()




