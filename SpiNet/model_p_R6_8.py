"""
This code will create the model described in our following paper
MoDL: Model-Based Deep Learning Architecture for Inverse Problems
by H.K. Aggarwal, M.P. Mani, M. Jacob from University of Iowa.

Paper dwonload  Link:     https://arxiv.org/abs/1712.02862

This function uses Lp norm with x^{k-1} as x0 for MM of next block
@author: haggarwal
"""
import tensorflow as tf
import numpy as np
from os.path import expanduser
home = expanduser("~")
epsilon=1e-5
TFeps=tf.constant(1e-5,dtype=tf.float32)


# function c2r contatenate complex input as new axis two two real inputs
c2r=lambda x:tf.stack([tf.real(x),tf.imag(x)],axis=-1)
#r2c takes the last dimension of real input and converts to complex
r2c=lambda x:tf.complex(x[...,0],x[...,1])

def createLayer(x, szW, trainning,lastLayer):
    """
    This function create a layer of CNN consisting of convolution, batch-norm,
    and ReLU. Last layer does not have ReLU to avoid truncating the negative
    part of the learned noise and alias patterns.
    """
    W=tf.get_variable('W',shape=szW,initializer=tf.contrib.layers.xavier_initializer())
    x = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    xbn=tf.layers.batch_normalization(x,training=trainning,fused=True,name='BN')

    if not(lastLayer):
        return tf.nn.relu(xbn)
    else:
        return xbn

def dw(inp,trainning,nLay):
    """
    This is the Dw block as defined in the Fig. 1 of the MoDL paper
    It creates an n-layer (nLay) residual learning CNN.
    Convolution filters are of size 3x3 and 64 such filters are there.
    nw: It is the learned noise
    dw: it is the output of residual learning after adding the input back.
    """
    lastLayer=False
    nw={}
    nw['c'+str(0)]=inp
    szW={}
    szW = {key: (3,3,64,64) for key in range(2,nLay)}
    szW[1]=(3,3,2,64)
    szW[nLay]=(3,3,64,2)

    for i in np.arange(1,nLay+1):
        if i==nLay:
            lastLayer=True
        with tf.variable_scope('Layer'+str(i)):
            nw['c'+str(i)]=createLayer(nw['c'+str(i-1)],szW[i],trainning,lastLayer)

    with tf.name_scope('Residual'):
        shortcut=tf.identity(inp)
        dw=shortcut+nw['c'+str(nLay)]
    return dw

	



class Aclass:
    """
    This class is created to do the data-consistency (DC) step as described in paper.
    """
    def __init__(self, csm,mask,lam,atb,p):
        with tf.name_scope('Ainit'):
            s=tf.shape(mask)
            self.nrow,self.ncol=s[0],s[1]
            self.pixels=self.nrow*self.ncol
            self.mask=mask
            self.csm=csm
            self.SF=tf.complex(tf.sqrt(tf.to_float(self.pixels) ),0.)
            self.lam = lam
            self.atb = r2c(atb)
            self.p = tf.complex(p-1,0.)
            self.eps = 1e-3
            #self.cgIter=cgIter
            #self.tol=tol
    def myAtA(self,img):
        with tf.name_scope('AtA'):
            coilImages=self.csm*img
            kspace=  tf.fft2d(coilImages)/self.SF
            temp=kspace*self.mask
            coilImgs =tf.ifft2d(temp)*self.SF
            coilComb= tf.reduce_sum(coilImgs*tf.conj(self.csm),axis=0)
            coilComb=coilComb+self.W*img
        return coilComb
	
    def myW(self,x,z):
        with tf.name_scope('W'):
            #self.W = self.lam*tf.pow(tf.conj(x-z)*(x-z)+self.eps,0.5*self.p) # equal to |x-z|^(p/2 -1)
            self.W = self.lam*tf.pow(tf.conj(x-z)*(x-z)+self.eps,self.p) # equal to |x-z|^(p/2 -1)
			# note that self.p <---- p-1  and p is initialized as p
    
    def myRhs(self,z):
        with tf.name_scope('rhs'):
            rhs = self.atb + self.W*z
        return rhs
		

def myCG(A,z,x0):
    """
    This is my implementation of CG algorithm in tensorflow that works on
    complex data and runs on GPU. It takes the class object as input.
    """
    z=r2c(z)
    x0 = r2c(x0)
    cond=lambda i,rTr,*_: tf.logical_and( tf.less(i,4), rTr>1e-10)
    def body(i,rTr,x_dummy,r,p):
        with tf.name_scope('cgBody'):
            Ap=A.myAtA(p)
            alpha = rTr / tf.to_float(tf.reduce_sum(tf.conj(p)*Ap))
            alpha=tf.complex(alpha,0.)
            x_dummy = x_dummy + alpha * p
            r = r - alpha * Ap
            rTrNew = tf.to_float( tf.reduce_sum(tf.conj(r)*r))
            beta = rTrNew / rTr
            beta=tf.complex(beta,0.)
            p = r + beta * p
        return i+1,rTrNew,x_dummy,r,p
    
    cond_outer = lambda j,x : tf.less(j,4)
    def body_outer(j,x):
        with tf.name_scope('Outerloop'):
            A.myW(x,z)
            #x_dummy=tf.zeros_like(z)
            x_dummy = x
            rhs = A.myRhs(z) - A.myAtA(x_dummy)
            i,r,p=0,rhs,rhs
            rTr = tf.to_float( tf.reduce_sum(tf.conj(r)*r),)
            loopVar=i,rTr,x_dummy,r,p
            x=tf.while_loop(cond,body,loopVar,name='CGwhile_in',parallel_iterations=1)[2]	
        return j+1,x
		
    #x=tf.zeros_like(z)
    x = x0
    j = 0
    loopVar_out = j,x
    x=tf.while_loop(cond_outer,body_outer,loopVar_out,name='CGwhile_out',parallel_iterations=1)[1]
	
    return c2r(x)

def getLambda():
    """
    create a shared variable called lambda.
    """
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        lam = tf.get_variable(name='lam1', dtype=tf.float32, initializer=.05)
    return lam


def getP():
    """
	Here p is 
	"""
    
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        p = tf.get_variable(name='p1', dtype=tf.float32, initializer=0.4)
    return p


def callCG(rhs):
    """
    this function will call the function myCG on each image in a batch
    """
    G=tf.get_default_graph()
    getnext=G.get_operation_by_name('getNext')
    _,_,csm,mask=getnext.outputs
    l=getLambda()
    l2=tf.complex(l,0.)
    def fn(tmp):
        c,m,r=tmp
        Aobj=Aclass(c,m,l2)
        y=myCG(Aobj,r)
        return y
    inp=(csm,mask,rhs)
    rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn2' )
    return rec

@tf.custom_gradient
def dcManualGradient(x):
    """
    This function impose data consistency constraint. Rather than relying on
    TensorFlow to calculate the gradient for the conjuagte gradient part.
    We can calculate the gradient manually as well by using this function.
    Please see section III (c) in the paper.
    """
    y=callCG(x)
    def grad(inp):
        out=callCG(inp)
        return out
    return y,grad


def dc(atb,z,x0,csm,mask,lam1,p):
    """
    This function is called to create testing model. It apply CG on each image
    in the batch.
    """
    lam2=tf.complex(lam1,0.)
    def fn( tmp ):
        c,m,atb,z,x0=tmp
        Aobj=Aclass( c,m,lam2,atb,p)
        y=myCG(Aobj,z,x0)
        return y
    inp=(csm,mask,atb,z,x0)
    rec=tf.map_fn(fn,inp,dtype=tf.float32,name='mapFn' )
    return rec

def makeModel(atb,csm,mask,training,nLayers,K,gradientMethod):
    """
    This is the main function that creates the model.

    """
    out={}
    out['dc0']=atb
    with tf.name_scope('myModel'):
        with tf.variable_scope('Wts',reuse=tf.AUTO_REUSE):
            for i in range(1,K+1):
                j=str(i)
                out['dw'+j]=dw(out['dc'+str(i-1)],training,nLayers)
                lam1=getLambda()
                p = getP()
                if gradientMethod=='AG':
                    out['dc'+j]=dc(atb,out['dw'+j],out['dc'+str(i-1)],csm,mask,lam1,p)
                elif gradientMethod=='MG':
                    if training:
                        out['dc'+j]=dcManualGradient(rhs)
                    else:
                        out['dc'+j]=dc(rhs,csm,mask,lam1)
    return out