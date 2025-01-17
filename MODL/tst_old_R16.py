"""
Created on Fri Mar 30 12:39:26 2018
This is the demo code. That should run without making any changes.
Please ensure that demoImage.hdf5 is in the same directory as this file tstDemo.py.

This code will load the learned model from the subdirectory 'savedModels'

This test code will load an  image for  from the demoImage.hdf5 file.

@author: haggarwal
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import supportingFunctions_old_R16 as sf
import model_old as mm
import scipy.io as sio
import h5py as h5py
from tqdm import tqdm
cwd=os.getcwd()
os.environ["CUDA_VISIBLE_DEVICES"]="1"


cwd=os.getcwd()
tf.reset_default_graph()

#%% choose a model from savedModels directory
nLayers=5
epochs=50
gradientMethod='AG'
K=10
#sigma=0.00
#subDirectory='14Mar_1105pm'
subDirectory='04Jul_1157pm_5L_10K_100E_AG_1BS'
#%%Read the testing data from dataset.hdf5 file

#tstOrg is the original ground truth
#tstAtb: it is the aliased/noisy image
#tstCsm: this is coil sensitivity maps
#tstMask: it is the undersampling mask

tstOrg,tstAtb,tstCsm,tstMask,_=sf.getTestingData(sigma = 0.01)
batchSize = 1

R = tstMask.shape[0]*tstMask.shape[1]*tstMask.shape[2]/np.sum(tstMask)
print('R = ',R)
#you can also read more testing data from dataset.hdf5 (see readme) file using the command
#tstOrg,tstAtb,tstCsm,tstMask=sf.getData('testing',num=100)

#%% Load existing model. Then do the reconstruction
nTst=tstOrg.shape[0]
nBatch= int(np.floor(np.float32(nTst)/batchSize))
nSteps= nBatch

print ('Now loading the model ...')


modelDir= cwd+'/savedModelsR16/'+subDirectory #complete path
rec=np.empty(tstAtb.shape,dtype=np.complex64) #rec variable will have output

tf.reset_default_graph()
loadChkPoint=tf.train.latest_checkpoint(modelDir)
config = tf.ConfigProto()
config.gpu_options.allow_growth=True

#csmT = tf.placeholder(tf.complex64,shape=(None,1,320,320),name='csmT')
#maskT= tf.placeholder(tf.complex64,shape=(None,320,320),name='maskT')
#atbT = tf.placeholder(tf.float32,shape=(None,320,320,2),name='atbT')
rec1 = []
csmT = tf.placeholder(tf.complex64,shape=(None,12,256,232),name='csm')
maskT= tf.placeholder(tf.complex64,shape=(None,256,232),name='mask')
atbT = tf.placeholder(tf.float32,shape=(None,256,232,2),name='atb')
#orgT = tf.placeholder(tf.float32,shape=(None,None,None,2),name='org')
out=mm.makeModel(atbT,csmT,maskT,False,nLayers,K,gradientMethod)
predT=out
saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
with tf.Session(config=config) as sess:
    saver.restore(sess,loadChkPoint)
    for step in tqdm(range(nSteps)):
        dataDict={atbT:tstAtb[step].reshape(-1,256,232,2),maskT:tstMask[step].reshape(-1,256,232),csmT:tstCsm[step].reshape(-1,12,256,232) }
        rec1.append(sess.run(predT,feed_dict=dataDict))
#rec=sf.r2c(rec.squeeze())
#print('Reconstruction done')

#%% normalize the data for calculating PSNR

#print('Now calculating the PSNR (dB) values')
#
#normOrg=sf.normalize01( np.abs(tstOrg))
#normAtb=sf.normalize01( np.abs(sf.r2c(tstAtb)))
#normRec=sf.normalize01(np.abs(rec))
#
#psnrAtb=sf.myPSNR(normOrg,normAtb)
#psnrRec=sf.myPSNR(normOrg,normRec)
#
#print ('*****************')
#print ('  ' + 'Noisy ' + 'Recon')
#print ('  {0:.2f} {1:.2f}'.format(psnrAtb,psnrRec))
#print ('*****************')

#%% Display the output images
#plot= lambda x: plt.imshow(x,cmap=plt.cm.gray, clim=(0.0, .8))
#plt.clf()
#plt.subplot(141)
#plot(np.fft.fftshift(tstMask[0]))
#plt.axis('off')
#plt.title('Mask')
#plt.subplot(142)
#plot(normOrg)
#plt.axis('off')
#plt.title('Original')
#plt.subplot(143)
#plot(normAtb)
#plt.title('Input, PSNR='+str(psnrAtb.round(2))+' dB' )
#plt.axis('off')
#plt.subplot(144)
#plot(normRec)
#plt.title('Output, PSNR='+ str(psnrRec.round(2)) +' dB')
#plt.axis('off')
#plt.subplots_adjust(left=0, right=1, top=1, bottom=0,wspace=.01)
#plt.show()


#a_dict = {'field1':tstOrg, 'field2': rec}
#sio.savemat('recon50_9.mat', {'a_dict': a_dict})
#path = './recons_h5/'
#savename = path+ 'recon_pat16V2_20.h5'
#hf = h5py.File(savename, 'w')
#hf.create_dataset('field1', data=tstOrg)
#hf.create_dataset('field2', data=rec)
#hf.close()
psnr_mean = []
for i in range(K+1):
    psnr = []
    rec2 = []
    for j in range(nTst):
        rec2.append(sf.normalize01(np.abs(sf.r2c(rec1[j]['dc'+str(i)]))))
        psnr.append(sf.myPSNR(normOrg[j,:,:],rec2[j]))
    psnr_mean.append(np.mean(psnr))
print(psnr_mean)
#    
#    
    
recon = rec2    
#%%
psnr_mean2 = []
for i in range(1,K+1):
    psnr_2 = []
    rec2 = []
    for j in range(nTst):
        rec2.append(sf.normalize01(np.abs(sf.r2c(rec1[j]['dw'+str(i)]))))
        psnr_2.append(sf.myPSNR(normOrg[j,:,:],rec2[j]))
    psnr_mean2.append(np.mean(psnr_2))
print(psnr_mean2)
#%%
#
#    
#psnr_mean2 = []
#for i in range(1,11):
#    psnr = []
#    rec3 = []
#    for j in range(164):
#        rec3.append(sf.normalize01(np.abs(sf.r2c(rec1[j]['dw'+str(i)]))))
#        psnr.append(sf.myPSNR(normOrg[j,:,:],rec3[j]))
#    psnr_mean2.append(np.mean(psnr))
    
    
#%%
#x = range(10)
#
#plt.plot(x,psnr_mean2)
#plt.xlabel('Iteration Number')
#plt.ylabel('PSNR  dB')

#%%
save_dir = 'save_recon_new/'
save_file_name = save_dir + 'MODL_psnr_recon_R16_K' +str(K)+'_new.mat'
sio.savemat(save_file_name,{'dc':np.asarray(psnr),'dw':np.asarray(psnr_2),'recon':np.asarray(recon),'org':np.asarray(normOrg)})
