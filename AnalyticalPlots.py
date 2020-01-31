# ----------------------------------------------------------------- #
# Preambule
# ----------------------------------------------------------------- #

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
import matplotlib as mpl
import sys
from tqdm import tqdm

# ----------------------------------------------------------------- #
# Input parameters
# ----------------------------------------------------------------- #

STD = 1

# Parameters truth
aS = 1
aT = 1
bS = 1
bT = 1
cS = 1
cT = 1
pars_true = [aS,aT,bS,bT,cS,cT]

# Parameters seen by sender
aS_s = [aS,0]
aT_s = [aT,STD]
bS_s = [bS,0]
bT_s = [bT,STD]
cS_s = [cS,STD]
cT_s = [cT,STD]
pars_s = [aS_s,aT_s,bS_s,bT_s,cS_s,cT_s]

# Parameters seen by target
aS_t = [aS,STD]
aT_t = [aT,0]
bS_t = [bS,STD]
bT_t = [bT,0]
cS_t = [cS,STD]
cT_t = [cT,STD]
pars_t = [aS_t,aT_t,bS_t,bT_t,cS_t,cT_t]

# ----------------------------------------------------------------- #
# Define functions
# ----------------------------------------------------------------- #

def p3(pars):
    aS,aT,bS,bT,cS,cT = pars
    return 1-scipy.stats.norm(0, np.sqrt(cS[1]**2+bS[1]**2)).cdf(-1-bS[0]+cS[0])

def Es3(pars):
    aS,aT,bS,bT,cS,cT = pars
    p3_ = p3(pars)
    return [p3_*(-cS[0]) + (1-p3_)*(-1-bS[0]),np.sqrt(cS[1]**2+bS[1]**2)]

def Et3(pars):
    aS,aT,bS,bT,cS,cT = pars
    p3_ = p3(pars)
    return [p3_*(-cT[0]) + (1-p3_)*(1+bT[0]),np.sqrt(cT[1]**2+bT[1]**2)]

def p2(pars):
    aS,aT,bS,bT,cS,cT = pars
    Et3_ = Et3(pars)
    return 1-scipy.stats.norm(0, np.sqrt(aT[1]**2+Et3_[1]**2)).cdf(-aT[0]-Et3_[0])

def Es2(pars):
    aS,aT,bS,bT,cS,cT = pars
    p2_ = p2(pars)
    Es3_ = Es3(pars)
    return [p2_*(Es3_[0])+(1-p2_)*(aS[0]),np.sqrt(aS[1]**2+Es3_[1]**2)]

def Et2(pars):
    aS,aT,bS,bT,cS,cT = pars
    p2_ = p2(pars)
    Et3_ = Et3(pars)
    return [p2_*(Et3_[0])+(1-p2_)*(-aT[0]),np.sqrt(aT[1]**2+Et3_[1]**2)]

def p1(pars):
    aS,aT,bS,bT,cS,cT = pars
    Es2_ = Es2(pars)
    return 1-scipy.stats.norm(0, Es2_[1]).cdf(-1-Es2_[0])

def EndBranches():
    print('p1                  :',np.round(p1(pars_s),3))
    print('p2                  :',np.round(p2(pars_t),3))
    print('p3                  :',np.round(p3(pars_s),3))
    print('')
    print('Status quo          :',np.round(1-p1(pars_s),3))
    print('Concession target   :',np.round(p1(pars_s)*(1-p2(pars_t)),3))
    print('Backing down sender :',np.round(p1(pars_s)*p2(pars_t)*(1-p3(pars_s)),3))
    print('Sanction imposed    :',np.round(p1(pars_s)*p2(pars_t)*(p3(pars_s)),3))

#%%
# ----------------------------------------------------------------- #
# Parameter sensitivity
# vary standard deviation of cs from 0 to 10
# vary mean value of cs from 0 to 10
#
# Assumptions:
# - STD/mean of ct is equal to that of cs
# - the STDs/means of ct and cs are the same for both parties
# ----------------------------------------------------------------- #

sig     = np.linspace(0.01,5,15)
mean    = np.linspace(-2,10,15)
p1s     = np.zeros(shape=(len(sig),len(mean)))
p2s     = np.zeros(shape=(len(sig),len(mean)))
p3s     = np.zeros(shape=(len(sig),len(mean)))

p1t     = np.zeros(shape=(len(sig),len(mean)))
p2t     = np.zeros(shape=(len(sig),len(mean)))
p3t     = np.zeros(shape=(len(sig),len(mean)))

p2t_vs = []

for i in tqdm(range(len(sig)),file=sys.stdout):
    for j in range(len(mean)):
        m = mean[j]
        s = sig[i]
        cS_s = [m,s]
        cT_s = [m,s]
        cS_t = [m,s]
        cT_t = [m,s]
        pars_s = [aS_s,aT_s,bS_s,bT_s,cS_s,cT_s]
        pars_t = [aS_t,aT_t,bS_t,bT_t,cS_t,cT_t]
        
        p1s[i,j] = p1(pars_s)
        p2s[i,j] = p2(pars_s)
        p3s[i,j] = p3(pars_s)
        
        p1t[i,j] = p1(pars_t)
        p2t[i,j] = p2(pars_t)
        p3t[i,j] = p3(pars_t)

#%%
# ----------------------------------------------------------------- #
# Plot
# ----------------------------------------------------------------- #

bords = np.arange(0,1.1,0.1)
cmap = plt.cm.rainbow

fig,((ax1,ax2,ax3),(ax4,ax5,ax6)) = plt.subplots(2,3,figsize=(12,8),sharex=True,sharey=True)

axes=[ax1,ax2,ax3,ax4,ax5,ax6]

c=ax1.contourf(mean,sig,p1s,bords,vmin=0,vmax=1,cmap=cmap)
ax2.contourf(mean,sig,p2s,bords,vmin=0,vmax=1,cmap=cmap)
ax3.contourf(mean,sig,p3s,bords,vmin=0,vmax=1,cmap=cmap)
ax4.contourf(mean,sig,p1t,bords,vmin=0,vmax=1,cmap=cmap)
ax5.contourf(mean,sig,p2t,bords,vmin=0,vmax=1,cmap=cmap)
ax6.contourf(mean,sig,p3t,bords,vmin=0,vmax=1,cmap=cmap)

c1=ax1.contour(mean,sig,p1s,bords,colors='k',zorder=1e3)
c2=ax2.contour(mean,sig,p2s,bords,colors='k',zorder=1e3)
c3=ax3.contour(mean,sig,p3s,bords,colors='k',zorder=1e3)
c4=ax4.contour(mean,sig,p1t,bords,colors='k',zorder=1e3)
c5=ax5.contour(mean,sig,p2t,bords,colors='k',zorder=1e3)
c6=ax6.contour(mean,sig,p3t,bords,colors='k',zorder=1e3)

ax1.clabel(c1, inline=1, fmt='%1.1f',fontsize=8)
ax2.clabel(c2, inline=1, fmt='%1.1f',fontsize=8)
ax3.clabel(c3, inline=1, fmt='%1.1f',fontsize=8)
ax4.clabel(c4, inline=1, fmt='%1.1f',fontsize=8)
ax5.clabel(c5, inline=1, fmt='%1.1f',fontsize=8)
ax6.clabel(c6, inline=1, fmt='%1.1f',fontsize=8)

ax4.set_xlabel(r'True $c_s$ and $c_t$')
ax5.set_xlabel(r'True $c_s$ and $c_t$')
ax6.set_xlabel(r'True $c_s$ and $c_t$')
ax1.set_ylabel(r'Standard deviation of $c_s$ and $c_t$')
ax4.set_ylabel(r'Standard deviation of $c_s$ and $c_t$')

for ax in axes:
    ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]])
    ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])
    ax.plot([1,1],[-1e3,1e3],'tomato',zorder=1e4)

ax1.text(0.05,0.92,r'(a) $p_1^s$',transform = ax1.transAxes,fontsize=13)
ax2.text(0.05,0.92,r'(b) $p_2^s$',transform = ax2.transAxes,fontsize=13)
ax3.text(0.05,0.92,r'(c) $p_3^s$',transform = ax3.transAxes,fontsize=13)

ax4.text(0.05,0.92,r'(d) $p_1^t$',transform = ax4.transAxes,fontsize=13)
ax5.text(0.05,0.92,r'(e) $p_2^t$',transform = ax5.transAxes,fontsize=13)
ax6.text(0.05,0.92,r'(f) $p_3^t$',transform = ax6.transAxes,fontsize=13)

axc     = fig.add_axes([0.22, -0.01, 0.6, 0.025])
norm    = mpl.colors.Normalize(vmin=0, vmax=1)
cb1     = mpl.colorbar.ColorbarBase(axc,norm=norm,extend='both',orientation='horizontal',cmap=cmap)
cb1.set_label('Probabilities',fontsize=15)
axc.tick_params(labelsize=13)

fig.tight_layout()
#plt.savefig('/Users/mmdekker/Documents/Werk/Figures/EconomicSanctions/Probabilities.png',dpi=200,bbox_inches='tight')

#%%
# ----------------------------------------------------------------- #
# Par sens for new plots
# ----------------------------------------------------------------- #


sig     = np.linspace(0.01,5,100)
mean    = np.linspace(-2,10,100)
p1s     = np.zeros(shape=(len(sig),len(mean)))
p2s     = np.zeros(shape=(len(sig),len(mean)))
p3s     = np.zeros(shape=(len(sig),len(mean)))

p1t     = np.zeros(shape=(len(sig),len(mean)))
p2t     = np.zeros(shape=(len(sig),len(mean)))
p3t     = np.zeros(shape=(len(sig),len(mean)))

p2t_vs = []
p3s_vs = []

STD = 1

# Parameters seen by sender
aS_s = [1,0]
aT_s = [1,STD]
bS_s = [1,0]
bT_s = [1,STD]
cS_s = [1,STD]
cT_s = [1,STD]

# Parameters seen by target
aS_t = [1,STD]
aT_t = [1,0]
bS_t = [1,STD]
bT_t = [1,0]
cS_t = [1,STD]
cT_t = [1,STD]

# Panel I
p1s     = np.zeros(shape=(len(sig),len(mean)))
p2s     = np.zeros(shape=(len(sig),len(mean)))
p3s     = np.zeros(shape=(len(sig),len(mean)))
p1t     = np.zeros(shape=(len(sig),len(mean)))
p2t     = np.zeros(shape=(len(sig),len(mean)))
p3t     = np.zeros(shape=(len(sig),len(mean)))
for i in tqdm(range(len(sig)),file=sys.stdout):
    for j in range(len(mean)):
        m = mean[j]
        s = sig[i]
        cS_s = [1,1]
        cS_t = [1,1]
        cT_s = [m,s]
        cT_t = [m,s]
        aS_s = [1,0]
        aS_t = [1,1]
        bS_s = [1,0]
        bS_t = [1,1]
            
        pars_s = [aS_s,aT_s,bS_s,bT_s,cS_s,cT_s]
        pars_t = [aS_t,aT_t,bS_t,bT_t,cS_t,cT_t]
        
        p1s[i,j] = p1(pars_s)
        p2s[i,j] = p2(pars_s)
        p3s[i,j] = p3(pars_s)
        
        p1t[i,j] = p1(pars_t)
        p2t[i,j] = p2(pars_t)
        p3t[i,j] = p3(pars_t)
p2t_vs.append(1-p2t)
    
# Panel II
p1s     = np.zeros(shape=(len(sig),len(mean)))
p2s     = np.zeros(shape=(len(sig),len(mean)))
p3s     = np.zeros(shape=(len(sig),len(mean)))
p1t     = np.zeros(shape=(len(sig),len(mean)))
p2t     = np.zeros(shape=(len(sig),len(mean)))
p3t     = np.zeros(shape=(len(sig),len(mean)))
for i in tqdm(range(len(sig)),file=sys.stdout):
    for j in range(len(mean)):
        m = mean[j]
        s = sig[i]
        cS_s = [1,1]
        cS_t = [1,1]        
        cT_s = [1,1]
        cT_t = [1,1]
        aS_s = [1,0]
        aS_t = [1,1]
        bS_s = [m,0]
        bS_t = [m,s]
            
        pars_s = [aS_s,aT_s,bS_s,bT_s,cS_s,cT_s]
        pars_t = [aS_t,aT_t,bS_t,bT_t,cS_t,cT_t]
        
        p1s[i,j] = p1(pars_s)
        p2s[i,j] = p2(pars_s)
        p3s[i,j] = p3(pars_s)
        
        p1t[i,j] = p1(pars_t)
        p2t[i,j] = p2(pars_t)
        p3t[i,j] = p3(pars_t)
p2t_vs.append(1-p2t)

# Panel III
p1s     = np.zeros(shape=(len(sig),len(mean)))
p2s     = np.zeros(shape=(len(sig),len(mean)))
p3s     = np.zeros(shape=(len(sig),len(mean)))
p1t     = np.zeros(shape=(len(sig),len(mean)))
p2t     = np.zeros(shape=(len(sig),len(mean)))
p3t     = np.zeros(shape=(len(sig),len(mean)))
for i in tqdm(range(len(sig)),file=sys.stdout):
    for j in range(len(mean)):
        m = mean[j]
        s = sig[i]
        cS_s = [m,s]
        cS_t = [m,s]
        cT_s = [1,1]
        cT_t = [1,1]
        aS_s = [1,0]
        aS_t = [1,1]
        bS_s = [1,0]
        bS_t = [1,1]
            
        pars_s = [aS_s,aT_s,bS_s,bT_s,cS_s,cT_s]
        pars_t = [aS_t,aT_t,bS_t,bT_t,cS_t,cT_t]
        
        p1s[i,j] = p1(pars_s)
        p2s[i,j] = p2(pars_s)
        p3s[i,j] = p3(pars_s)
        
        p1t[i,j] = p1(pars_t)
        p2t[i,j] = p2(pars_t)
        p3t[i,j] = p3(pars_t)
p3s_vs.append(p3s)

# Panel VI
p1s     = np.zeros(shape=(len(sig),len(mean)))
p2s     = np.zeros(shape=(len(sig),len(mean)))
p3s     = np.zeros(shape=(len(sig),len(mean)))
p1t     = np.zeros(shape=(len(sig),len(mean)))
p2t     = np.zeros(shape=(len(sig),len(mean)))
p3t     = np.zeros(shape=(len(sig),len(mean)))
for i in tqdm(range(len(sig)),file=sys.stdout):
    for j in range(len(mean)):
        m = mean[j]
        s = sig[i]
        cS_s = [1,1]
        cS_t = [1,1]        
        cT_s = [1,1]
        cT_t = [1,1]
        aS_s = [1,0]
        aS_t = [1,1]
        bS_s = [m,s]
        bS_t = [m,1]
        
        pars_s = [aS_s,aT_s,bS_s,bT_s,cS_s,cT_s]
        pars_t = [aS_t,aT_t,bS_t,bT_t,cS_t,cT_t]
        
        p1s[i,j] = p1(pars_s)
        p2s[i,j] = p2(pars_s)
        p3s[i,j] = p3(pars_s)
        
        p1t[i,j] = p1(pars_t)
        p2t[i,j] = p2(pars_t)
        p3t[i,j] = p3(pars_t)
p3s_vs.append(p3s)

#%%
# ----------------------------------------------------------------- #
# New plots
# ----------------------------------------------------------------- #

bords = np.arange(0,1.1,0.1)
cmap = plt.cm.rainbow

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)

axes=[ax1,ax2]

c=ax1.contourf(mean,sig,p2t_vs[0],bords,vmin=0,vmax=1,cmap=cmap)
ax2.contourf(mean,sig,p2t_vs[1],bords,vmin=0,vmax=1,cmap=cmap)

c1=ax1.contour(mean,sig,p2t_vs[0],bords[1:],colors='k',zorder=1e3)
c2=ax2.contour(mean,sig,p2t_vs[1],bords[1:],colors='k',zorder=1e3)

ax1.clabel(c1, inline=1, fmt='%1.1f',fontsize=7)
ax2.clabel(c2, inline=1, fmt='%1.1f',fontsize=7)

ax1.set_ylabel(r'Standard deviation of $c_t$')
ax2.set_ylabel(r'Standard deviation of $b_s$')
ax1.set_xlabel(r'True $c_t$ (ceteris paribus)')
ax2.set_xlabel(r'True $b_s$ (ceteris paribus)')

for ax in axes:
    ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]])
    ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])
    ax.plot([1,1],[-1e3,1e3],'k:',zorder=1e4)

ax1.text(0.05,0.88,r'(a) $1-p_2^t$',transform = ax1.transAxes,fontsize=13,bbox = dict(boxstyle='round',lw=1, facecolor='w',alpha=1),zorder=1e9)
ax2.text(0.05,0.88,r'(b) $1-p_2^t$',transform = ax2.transAxes,fontsize=13,bbox = dict(boxstyle='round',lw=1, facecolor='w',alpha=1),zorder=1e9)

axc     = fig.add_axes([0.22, -0.01, 0.6, 0.025])
norm    = mpl.colors.Normalize(vmin=0, vmax=1)
cb1     = mpl.colorbar.ColorbarBase(axc,norm=norm,orientation='horizontal',cmap=cmap)
cb1.set_label('Probabilities',fontsize=15)
axc.tick_params(labelsize=13)

fig.tight_layout()
plt.savefig('/Users/mmdekker/Documents/Werk/Figures/EconomicSanctions/Probabilities_V1.png',dpi=200,bbox_inches='tight')

# Second plot

fig,(ax1,ax2) = plt.subplots(1,2,figsize=(10,5),sharex=True,sharey=True)

axes=[ax1,ax2]

c=ax1.contourf(mean,sig,p3s_vs[0],bords,vmin=0,vmax=1,cmap=cmap)
ax2.contourf(mean,sig,p3s_vs[1],bords,vmin=0,vmax=1,cmap=cmap)

c1=ax1.contour(mean,sig,p3s_vs[0],bords[1:],colors='k',zorder=1e3)
c2=ax2.contour(mean,sig,p3s_vs[1],bords[1:],colors='k',zorder=1e3)

ax1.clabel(c1, inline=1, fmt='%1.1f',fontsize=7)
ax2.clabel(c2, inline=1, fmt='%1.1f',fontsize=7)

ax1.set_ylabel(r'Standard deviation of $c_s$')
ax2.set_ylabel(r'Standard deviation of $b_s$')
ax1.set_xlabel(r'True $c_s$ (ceteris paribus)')
ax2.set_xlabel(r'True $b_s$ (ceteris paribus)')

for ax in axes:
    ax.set_xlim([ax.get_xlim()[0],ax.get_xlim()[1]])
    ax.set_ylim([ax.get_ylim()[0],ax.get_ylim()[1]])
    ax.plot([1,1],[-1e3,1e3],'k:',zorder=1e4)

ax1.text(0.05,0.88,r'(a) $p_3^s$',transform = ax1.transAxes,fontsize=13,bbox = dict(boxstyle='round',lw=1, facecolor='w',alpha=1),zorder=1e9)
ax2.text(0.05,0.88,r'(b) $p_3^s$',transform = ax2.transAxes,fontsize=13,bbox = dict(boxstyle='round',lw=1, facecolor='w',alpha=1),zorder=1e9)

axc     = fig.add_axes([0.22, -0.01, 0.6, 0.025])
norm    = mpl.colors.Normalize(vmin=0, vmax=1)
cb1     = mpl.colorbar.ColorbarBase(axc,norm=norm,orientation='horizontal',cmap=cmap)
cb1.set_label('Probabilities',fontsize=15)
axc.tick_params(labelsize=13)

fig.tight_layout()
plt.savefig('/Users/mmdekker/Documents/Werk/Figures/EconomicSanctions/Probabilities_V2.png',dpi=200,bbox_inches='tight')
