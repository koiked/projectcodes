import sys
argvs=sys.argv
argc=len(argvs)
import numpy as np
import numpy.linalg as LA
import scipy as sp
import scipy.linalg as SI
if(argc !=2) 
    num = 9
else 
    num=open(argvs[1])
rarry=np.random.randint(0,2,(num,num)) # making random 10x10 matrix with value 1 or 0)< in here you may put your original matrix
ar2=np.triu(rarry-np.diag(np.diag(rarry))) # pick up upper triangle of random matrix with 0 diag
ar3=ar2+ar2.T # make symmetric matrix
ar4=np.diag(np.sum(ar3,axis=0)) #calc diagonal
ar5=ar4-ar3 # calc lapracian matrix ( ar5=2*ar4-ar3 Mr. kogai version)
u,v=SI.eigh(ar5) # calc eigen value u and egen vectors v 
print(ar5)
print(u)
print(v)

