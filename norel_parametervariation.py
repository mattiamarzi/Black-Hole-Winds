import numpy as np
import matplotlib.pyplot as plt
colors=['purple','xkcd:periwinkle','xkcd:grass green']

x0=[10,1E2,1E3]		             #r0 in units of R_S (I don't call it r0 because it isn't in cm) 
xi0=[3,4,5]			             #logaritmic scale
lambda_edd=[1E-2,1E-1,1]         #ratio between bolometric and eddington luminosities
alpha=[0,1,2]				     #density profile
k=10                             #proportion coefficient between bolometric and ionizing luminosities

nh=np.linspace(0,1E24,50)            #column density, units cm-2
v0=np.sqrt(np.divide(1,x0))          #vout=escape velocity in units of c
ratio=np.zeros([len(xi0),len(nh)])   #matrix in which we'll store the ratio between nh and n0r0

for i in range(len(alpha)):
    for i0 in range(len(lambda_edd)):
        plt.figure(figsize=(18,10))
        plt.suptitle('α={:1.0f}, $λ_{{Edd}}$={:3.2f}\n\n\n' .format(alpha[i], lambda_edd[i0]),x=0.51,fontsize=12)
        
        counter=1
        for i1 in range(len(x0)):
            plt.subplot(3,len(x0),counter) 
            plt.title('$r_0$={:3.0f} $R_S$\n  '.format(x0[i1]),fontsize=12)
            
            for i2 in range(len(xi0)):
                n0r0=lambda_edd[i0]*4.39*1E32/(k*(10**xi0[i2])*x0[i1])
                for i3 in range(len(nh)):
                    ratio[i2,i3]=nh[i3]/n0r0
                if alpha[i]==0:
                    delta_r=ratio[i2,:]
                elif alpha[i]==1:
                    delta_r=np.exp(ratio[i2,:])-1
                elif alpha[i]==2:
                    nh=np.linspace(0,1E21,50) #let's redefine nh to remove previous modifications
                    nh_limit=n0r0	
                    nh_ref=0 
                    for z in range(len(nh)):
                        if nh[z]>nh_limit:
                            if nh_ref==0: 
                                nh_ref=nh[z-1]
                            nh[z]=nh_ref
                    for i4 in range(len(nh)):
                        ratio[i2,i4]=nh[i4]/n0r0
                    delta_r=1/(1-ratio[i2,:])-1
                plt.loglog(nh,delta_r,color=colors[i2],label='$log(ξ_0)$={:2.1f}'.format(xi0[i2]))
            if counter == 1:
                plt.ylabel('$\Delta$r',fontsize=12)
                plt.legend(fontsize=12)
            plt.grid()
        
            plt.subplot(3,len(x0),counter+len(x0))
            for i2 in range(len(xi0)):
                if alpha[i]==0:
                    xi_r=np.log10(10**xi0[i2]*(ratio[i2,:]+1)**(-2))
                elif alpha[i]==1:
                    xi_r=np.log10(10**xi0[i2]*(np.exp(ratio[i2,:]))**(-1))
                elif alpha[i]==2:
                    xi_r=np.log10(10**xi0[i2]*(1/(1-ratio[i2,:]))**(0))
                plt.plot(nh,xi_r,color=colors[i2])
            if counter == 1:
                plt.ylabel('$ξ(r)$',fontsize=12)
            plt.grid()
        
            plt.subplot(3,len(x0),counter+len(x0)*2) #Let's do the third row
            for i2 in range(len(xi0)):
                if alpha[i]==0:
                    v_r=v0[i1]*(ratio[i2,:]+1)**(-1)
                if alpha[i]==1:
                    v_r=v0[i1]*(np.exp(ratio[i2,:]))**(-0.5)
                if alpha[i]==2:
                    v_r=v0[i1]*(1/(1-ratio[i2,:]))**(0)
                plt.plot(nh,v_r,color=colors[i2])
            if counter == 1:
                plt.ylabel('$v(r)$ (c)',fontsize=12)
            plt.grid()
            plt.xlabel(r'$N_H$ ($cm^{-2}$)',fontsize=12)
            counter+=1
        plt.savefig('Non Relativistic Parameters Variation α={:1.0f}, $λ_{{Edd}}$={:3.2f}.png'.format(alpha[i], lambda_edd[i0]))

plt.show()

