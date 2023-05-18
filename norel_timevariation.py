import numpy as np
import matplotlib.pyplot as plt
colors=['purple','xkcd:periwinkle','xkcd:grass green']

x0=[10,1E2,1E3]		             #r0 in units of R_S (I don't call it r0 because it isn't in cm) 
xi0=[3,4,5]			             #logaritmic scale
l_ion=1.3*1E43			         #erg/s
Mbh=1E8                          #Msun
alpha=[0,1,2]				     #density profile


v0=np.sqrt(np.divide(1,x0))                  #vout=escape velocity in units of c
time=np.linspace(0,1E6,5000)
r0=np.multiply(x0,Mbh*2.982*10**5)	         #converting x0 in cm
rmax=np.zeros([len(xi0),len(time+1)])        #matrix in which I'll store values of r of the top of the column
rmin=np.zeros([len(xi0),len(time+1)])        #matrix in which I'll store values of r of the bottom of the column
nh=np.zeros([len(xi0),len(time)])            #matrix in which I'll store the values of the column density
nh_limit=1E24                                #column density at which the wind breaks off

def r(t,vout,rout,alpha):
    r=(rout**((4-alpha)/2)+(4-alpha)/2*vout*(3*10**10)*rout**((2-alpha)/2)*t)**(2/(4-alpha))
    return r

for i in range(len(alpha)):
    plt.figure(figsize=(19,17))
    plt.suptitle('α={:1.0f}\n' .format(alpha[i]), y=0.93, x=0.51,fontsize=12)
    
    counter=1
    for i1 in range(len(r0)):
        plt.subplot(5,len(x0),counter) 
        plt.title('$r_0$={:3.0f} $R_S$\n  '.format(x0[i1]),fontsize=12)
        
        for i2 in range(len(xi0)):
            n0=l_ion/((r0[i1]**2)*10**xi0[i2])
            t_ref=0  #reference to find at which time nh will become greater than 1E24
            rmax[i2,0]=r0[i1]
            rmin[i2,0]=r0[i1]
            for i3 in range(len(time)):
                if alpha[i]==0:
                    nh[i2,i3]=n0*(rmax[i2,i3]-rmin[i2,i3])
                if alpha[i]==1:
                    nh[i2,i3]=n0*r0[i1]*np.log(rmax[i2,i3]/rmin[i2,i3])
                if alpha[i]==2:
                    nh[i2,i3]=n0*(r0[i1])**2*(1/rmin[i2,i3]-1/rmax[i2,i3])
                if i3<len(time)-1:  #in this way the index doesn't go out of the array
                    rmax[i2,i3+1]=r(time[i3+1],v0[i1],r0[i1],alpha[i])
                    if nh[i2,i3]<nh_limit:
                        if t_ref==0:  #in this way, once the wind breaks, even if nh decreases below critic value, rmin continues mooving
                            rmin[i2,i3+1]=r0[i1]
                    else:
                        if t_ref==0:  #in this way I understand when nh becomes greater than the critic value
                            t_ref=i3
                    if nh[i2,i3]>=nh_limit or t_ref!=0:
                        rmin[i2,i3+1]=r(time[i3+1-t_ref],v0[i1],r0[i1],alpha[i])
            plt.loglog(time,nh[i2,:],color=colors[i2],label='$log(ξ_0)$={:2.1f}'.format(xi0[i2]))
        if counter==1:
            plt.ylabel('$N_H(t)$',fontsize=12)
            plt.legend(fontsize=12)
        plt.grid()
        
        plt.subplot(5,len(x0),counter+len(x0))
        for i2 in range(len(xi0)):
            delta_r=(rmax[i2,:]-rmin[i2,:])/rmin[i2,:]
            plt.loglog(time,delta_r,color=colors[i2])
        if counter==1:
            plt.ylabel('$\Delta$r(t)',fontsize=12)
        plt.grid()
    
        plt.subplot(5,len(x0),counter+len(x0)*2)
        for i2 in range(len(xi0)):
            if alpha[i]==0:
                xi=np.log10(((10**xi0[i2])*(r0[i1])**2)/(rmax[i2,:]*rmin[i2,:]))
            if alpha[i]==1:
                xi=np.zeros(len(time))
                xi[0]=xi0[i2]
                for i3 in range(1,len(time)):  #I must eliminate i3=0, because I would have rmax=rmin and there would be a divergence
                    xi[i3]=np.log10(((10**xi0[i2])*r0[i1])*(1/rmin[i2,i3]-1/rmax[i2,i3])*1/np.log(rmax[i2,i3]/rmin[i2,i3]))
            if alpha[i]==2:
                xi=np.zeros(len(time))
                for i3 in range(len(time)):
                    xi[i3]=xi0[i2]
            plt.semilogx(time,xi,color=colors[i2])
        if counter==1:
            plt.ylabel('$ξ(t)$',fontsize=12)
        plt.grid()
        
        plt.subplot(5,len(x0),counter+len(x0)*3)
        for i2 in range(len(xi0)):
            vtop=v0[i1]*(rmax[i2,:]/r0[i1])**((alpha[i]-2)/2)
            plt.semilogx(time,vtop,color=colors[i2])
        if counter==1:
            plt.ylabel('$v_{top}(t)$ (c)',fontsize=12)
        plt.grid()
        
        plt.subplot(5,len(x0),counter+len(x0)*4)
        for i2 in range(len(xi0)):
            t_ref=0
            vbottom=np.zeros(len(time))
            for i3 in range(len(time)):
                if rmin[i2,i3]==r0[i1]:
                    vbottom[i3]=0
                else:
                    if t_ref==0:
                        t_ref=i3-1
                    vbottom[i3]=v0[i1]*(rmax[i2,i3-t_ref]/r0[i1])**((alpha[i]-2)/2)
            plt.semilogx(time,vbottom,color=colors[i2])
        if counter==1:
            plt.ylabel('$v_{bottom}(t)$ (c)',fontsize=12)
        plt.grid()
        plt.xlabel(r't (s)',fontsize=12)
        counter+=1
     
    #plt.savefig('Non Relativistic Time Variation, α={:1.0f}.png' .format(alpha[i]))
plt.show()