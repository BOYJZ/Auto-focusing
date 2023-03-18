import matplotlib.pyplot as plt

asnr=[1,2,3,5,10,100,1000]
GS=		[0.02,0.2,0.29,0.67,0.98,1,1]
PSO=		[0.01, 0.07, 0.06, 0.13, 0.37, 1.0, 1.0]
GPSO=		[0.01, 0.06, 0.15, 0.25, 0.57, 1.0, 1.0]
CNN=		[0.185,0.83,0.95,0.99,0.995,1.0,1.0]

plt.plot(asnr, GS, label='GS')
plt.plot(asnr, PSO, label='PSO')
plt.plot(asnr, GPSO, label='GPSO')
plt.plot(asnr, CNN, label='CNN')

plt.xscale('log')
plt.xlabel('log(snr)')
plt.ylabel('pro of success')
plt.title('Success Probability vs SNR')
plt.legend()
plt.show()

asnr=[1,1.5,2,3,5,10]
GS=[0.07,0.1,0.22,0.33,0.57,0.99]
CNN=[0.185,0.55,0.83,0.95,0.99,0.995]
plt.plot(asnr, GS, label='GS')
plt.plot(asnr, CNN, label='CNN')
plt.xlabel('small snr')
plt.ylabel('pro of success')
plt.title('Success Probability vs SNR')
plt.legend()
plt.show()
