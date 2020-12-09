import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import BayesianRidge
from scipy.signal import hilbert, chirp
from scipy.interpolate import interp1d
n_order = 5
reg = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
reg_2 = BayesianRidge(tol=1e-6, fit_intercept=False, compute_score=True)
dat=np.loadtxt("5717.dat",dtype='float')
size=dat.shape[0]
t=np.copy(dat[:,0])
x_test = np.linspace(t[0]-10, t[size-1]+10, 100)
it=np.zeros((size,4),dtype='float')
it2=np.zeros((size-1,4),dtype='float')
it3=np.zeros((size-2,4),dtype='float')
it4=np.zeros((size-2,1),dtype='float')
intensities=np.zeros((size,8),dtype='float')
ft=np.zeros((size,4),dtype='complex')
#print(np.max(x_test))
it[:,0]=dat[:,3]+dat[:,6]
it[:,1]=(dat[:,4]+dat[:,5])
it[:,2]=dat[:,3]-dat[:,6]
it[:,3]=-dat[:,4]+dat[:,5]
for i in range(size-1):
    it2[i,:]=it[i,:]-it[i+1,:]
for i in range(size-2):
    it3[i,:]=it2[i,:]/it2[i+1,:] 
    it4[i]=np.average(it3[i,:])
fig,ax3=plt.subplots(2,4,figsize=(8,8))
#ax3[:,:].set_ylim(-20, 20)
for i in range(4):
    ax3[0,i].plot(t[:size-1],it2[:,i])
    ax3[1,i].plot(t[:size-2],it3[:,i])
    ax3[1,i].plot(t[:size-2],it4)
    ax3[0,i].set_ylim(-60, 60)
    ax3[1,i].set_ylim(-60, 60)
fig,ax4=plt.subplots(2,3,figsize=(8,8))
#ax4[:,:].set_ylim(-20, 20)
k=0
for i in range(4):
    for j in range(i+1,4):
        ii=k//3
        jj=k%3
        ax4[ii,jj].plot(t[:size-1],it2[:,i]/it2[:,j])
        ax4[ii,jj].set_ylim(-60, 60)
        k+=1
#it[:,0]=dat[:,3]
#it[:,1]=dat[:,4]
#it[:,2]=dat[:,5]
#it[:,3]=dat[:,6]

ax_train=np.vander(t,n_order+1,increasing=True)
ax_test=np.vander(x_test,n_order+1,increasing=True)

#print(np.max(t))
for i in range(4):
    init = [1., 1e-3]
    tp=it[:,i]
    reg.set_params(alpha_init=init[0], lambda_init=init[1])
    reg_2.set_params(alpha_init=init[0], lambda_init=init[1])
    reg.fit(ax_train, tp)
    ymean, ystd = reg.predict(ax_test, return_std=True)
    ymean2=reg.predict(ax_train)
    asig=hilbert(tp-ymean2)
    aenv=np.where(asig>0,asig,0)
    not_nan=np.where(asig>0)
    indices=np.arange(len(aenv))
    aenv0=aenv[not_nan]
    t2=t[not_nan]
    ax_train2=np.vander(t2,n_order+1,increasing=True)
    #aenv=np.interp(indices,indices[not_nan],aenv[not_nan])
    #tfit=t[np.where(asig>0,True,False)]
    #print(tfit)
    f1=interp1d(t[not_nan],aenv[not_nan],kind='cubic',fill_value="extrapolate")
    #aenv=f1(t)
    aenv=f1(t)
    #print(len(aenv),len(tfit),len(tp))
    #aenv=np.abs(asig)
    reg_2.fit(ax_train2,aenv0)
    aenvest=reg_2.predict(ax_train)
    intensities[:,i]=ymean2
    ft[:,i]=np.fft.fft(tp-ymean2)
    #intensities[:,i+4]=aenvest
    intensities[:,i+4]=aenv
temp1=intensities[:,6]/intensities[:,7]
phip12=np.arctan(temp1)
intdash=intensities[:,6]*intensities[:,6]+intensities[:,7]*intensities[:,7]
intdash=np.sqrt(intdash)
temp1=intensities[:,4]/intdash
phim12=np.arcsin(temp1)
phi1=(phim12+phip12)/2
phi2=(phip12-phim12)/2
c21=np.cos(phim12+phip12)
c22=np.cos(-phim12+phip12)
s21=np.sin(phim12+phip12)
s22=np.sin(-phim12+phip12)
Ia2=intensities[:,2]
Ia3=intensities[:,3]
I1=-s22*Ia2+c22*Ia3
I2=-s21*Ia2+c21*Ia3
s2p=np.sin(2*phim12)
I1=I1/s2p
I2=I2/s2p
fig,axes=plt.subplots(1,4,figsize=(8,4))
for i, ax in enumerate(axes):
    tp=it[:,i]
    ymean2= intensities[:,i]
    aenvest=intensities[:,i+4]
    ax.scatter(t, tp, s=10, alpha=0.5, label="observation")
    ax.plot(t, tp-ymean2, label="observation2")
    ax.plot(t, aenvest, label="enverope")
    ax.plot(t, ymean2, color="red", label="predict mean")
    #ax.fill_between(x_test, ymean-ystd, ymean+ystd,
    #               color="pink", alpha=0.5, label="predict std")
    #print(reg.coef_)
    ax.set_ylim(-40, 150)
fig2,axes2=plt.subplots(4,1,figsize=(8,4))
axes2[0].plot(t,np.degrees(phi1),label="phi1")
axes2[0].plot(t,np.degrees(phi2),label='phi2')
#axes2[1].plot(t,intdash)
axes2[1].plot(t,-I2+I1)
axes2[1].plot(t,I2+I1)
axes2[1].plot(t,I2)
axes2[1].plot(t,I1)
axes2[1].plot(t,intensities[:,0])
axes2[1].set_ylim(-10, 150)
for i in range(4):
    axes2[2].plot(np.abs(ft[:,i]))
    axes2[3].plot(np.angle(ft[:,i]))
axes2[2].set_xlim(0,size/2)
axes2[3].set_xlim(0,size/2)
#axes2.plot(t,phim12)
#axes2.plot(t,phip12)
plt.tight_layout()
plt.show()
#print(dat.shape)