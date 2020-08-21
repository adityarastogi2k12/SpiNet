# -*- coding: utf-8 -*-
"""
This is the training code to train the model as described in the following article:

SpiNet: A Deep Neural Network for Schatten p-norm Regularized Medical Image Reconstruction
by Aditya Rastogi, Phaneendra Yalavarthy from Indian Institute of Sciences.


This code solves the following optimization problem:

    argmin_x ||Ax-b||_2^2 + ||x-Dw(x)||^p_p

 'A' can be any measurement operator. Here we consider parallel imaging problem in MRI where
 the A operator consists of undersampling mask, FFT, and coil sensitivity maps.

Dw(x): it represents the residual learning CNN.

Here is the description of the parameters that you can modify below.

epochs: how many times to pass through the entire dataset

nLayer: number of layers of the convolutional neural network.
        Each layer will have filters of size 3x3. There will be 64 such filters
        Except at the first and the last layer.

gradientMethod: MG or AG. set MG for 'manual gradient' of conjuagate gradient (CG) block
                as discussed in section 3 of the above paper. Set it to AG if
                you want to rely on the tensorflow to calculate gradient of CG.

K: it represents the number of iterations of the alternating strategy as
    described in Eq. 10 in the paper.  Also please see Fig. 1 in the above paper.
    Higher value will require a lot of GPU memory. Set the maximum value to 20
    for a GPU with 16 GB memory. Higher the value more is the time required in training.

sigma: the standard deviation of Gaussian noise to be added in the k-space

batchSize: You can reduce the batch size to 1 if the model does not fit on GPU.

Output:

After running the code the output model will be saved in the subdirectory 'savedModels'.
You can give the name of the generated ouput directory in the tstDemo.py to
run the newly trained model on the test data.


@author: Aditya Rastogi
"""

# import some librariesw
import os,time
os.environ["CUDA_VISIBLE_DEVICES"]="0"
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import numpy as np
import tensorflow as tf
from datetime import datetime
from tqdm import tqdm
import supportingFunctions_R16 as sf
import model_p_R6_8 as mm

tf.reset_default_graph()
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
#config.gpu_options.visible_device_list= '0,1' #see the gpu 0, 1, 2

#--------------------------------------------------------------
#% SET THESE PARAMETERS CAREFULLY
nLayers=5
epochs= 50
batchSize=8
gradientMethod='AG'
K=5
learn_rate = 7e-4
sigma=0.01
#%% to train the model with higher K values  (K>1) such as K=5 or 10,
# it is better to initialize with a pre-trained model with K=1.
restoreWeights = False
if K>1:
    restoreWeights=True
    restoreFromModel='10May_1101pm_5L_5K_100E_AG_8BS_lr_0.0007'

if restoreWeights:
    wts=sf.getWeights('savedModelsR16_8/'+restoreFromModel,chkPointNum='-100')
#--------------------------------------------------------------------------
#%%Generate a meaningful filename to save the trainined models for testing
print ('*************************************************')
start_time=time.time()
saveDir='savedModelsR16_8/'
cwd=os.getcwd()
directory=saveDir+datetime.now().strftime("%d%b_%I%M%P_")+ \
 str(nLayers)+'L_'+str(K)+'K_'+str(epochs)+'E_'+gradientMethod +'_'+str(batchSize)+'BS_lr_'+str(learn_rate)

if not os.path.exists(directory):
    os.makedirs(directory)
sessFileName= directory+'/model'
print('x0 for W of 1st iter of MM is init using x^{k-1} and so does x for init. CG')

#%% save test model
tf.reset_default_graph()
csmT = tf.placeholder(tf.complex64,shape=(None,12,256,232),name='csm')
maskT= tf.placeholder(tf.complex64,shape=(None,256,232),name='mask')
atbT = tf.placeholder(tf.float32,shape=(None,256,232,2),name='atb')

out=mm.makeModel(atbT,csmT,maskT,False,nLayers,K,gradientMethod)
predTst=out['dc'+str(K)]
predTst=tf.identity(predTst,name='predTst')
sessFileNameTst=directory+'/modelTst'

saver=tf.train.Saver()
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    savedFile=saver.save(sess, sessFileNameTst,latest_filename='checkpointTst')
print ('testing model saved:' +savedFile)
#%% read multi-channel dataset
trnOrg,trnAtb,trnCsm,trnMask=sf.getData('training')
trnOrg,trnAtb=sf.c2r(trnOrg),sf.c2r(trnAtb)

#%%
tf.reset_default_graph()
csmP = tf.placeholder(tf.complex64,shape=(None,None,None,None),name='csm')
maskP= tf.placeholder(tf.complex64,shape=(None,None,None),name='mask')
atbP = tf.placeholder(tf.float32,shape=(None,None,None,2),name='atb')
orgP = tf.placeholder(tf.float32,shape=(None,None,None,2),name='org')


#%% creating the dataset
nTrn=trnOrg.shape[0]
nBatch= int(np.floor(np.float32(nTrn)/batchSize))
nSteps= nBatch*epochs

with tf.device('/cpu:0'):
    trnData = tf.data.Dataset.from_tensor_slices((orgP,atbP,csmP,maskP))
    trnData = trnData.cache()
    trnData=trnData.repeat(count=epochs)
    trnData = trnData.shuffle(buffer_size=trnOrg.shape[0])
    trnData=trnData.batch(batchSize)
    trnData=trnData.prefetch(5)
    iterator=trnData.make_initializable_iterator()
    orgT,atbT,csmT,maskT = iterator.get_next('getNext')

#%% make training model

out=mm.makeModel(atbT,csmT,maskT,True,nLayers,K,gradientMethod)
predT=out['dc'+str(K)]
predT=tf.identity(predT,name='pred')
loss = tf.reduce_mean(tf.reduce_sum(tf.pow(predT-orgT, 2),axis=0))
tf.summary.scalar('loss', loss)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

with tf.name_scope('optimizer'):
    optimizer = tf.train.AdamOptimizer(learning_rate= learn_rate)
    gvs = optimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    opToRun=optimizer.apply_gradients(capped_gvs)


#%% training code


print ('training started at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('parameters are: Epochs:',epochs,' BS:',batchSize,'nSteps:',nSteps,'nSamples:',nTrn)

saver = tf.train.Saver(max_to_keep=100)
totalLoss,ep=[],0
lossT = tf.placeholder(tf.float32)
lossSumT = tf.summary.scalar("TrnLoss", lossT)

with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    if restoreWeights:
        sess=sf.assignWts(sess,nLayers,wts)

    feedDict={orgP:trnOrg,atbP:trnAtb, maskP:trnMask,csmP:trnCsm}
    sess.run(iterator.initializer,feed_dict=feedDict)
    savedFile=saver.save(sess, sessFileName)
    print("Model meta graph saved in::%s" % savedFile)

    writer = tf.summary.FileWriter(directory, sess.graph)
    for step in tqdm(range(nSteps)):
        try:
            tmp,_,_=sess.run([loss,update_ops,opToRun])
            totalLoss.append(tmp)
            if np.remainder(step+1,nBatch)==0:
                ep=ep+1
                avgTrnLoss=np.mean(totalLoss)
                lossSum=sess.run(lossSumT,feed_dict={lossT:avgTrnLoss})
                print(avgTrnLoss)
                writer.add_summary(lossSum,ep)
                totalLoss=[] #after each epoch empty the list of total loos
            if np.remainder(ep,5)==0:
                savedfile=saver.save(sess, sessFileName,global_step=ep,write_meta_graph=True)
                writer.close()

        except tf.errors.OutOfRangeError:
            break
        

end_time = time.time()
print ('Trianing completed in minutes ', ((end_time - start_time) / 60))
print ('training completed at', datetime.now().strftime("%d-%b-%Y %I:%M %P"))
print ('*************************************************')

#%%

