import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np

def conv(x, out_channel, kernel_size, stride=1, dilation=1):
    x = slim.conv2d(x, out_channel, kernel_size, stride, rate=dilation,activation_fn=None)
    return x

def global_avg_pool2D(x):
    with tf.variable_scope(None, 'global_pool2D'):
        n,h,w,c=x.get_shape().as_list
        x = slim.avg_pool2d(x, (h,w), stride=1)
    return x

def global_context_module(x,squeeze_depth,fuse_method='add',attention_method='att',scope=None):

    assert fuse_method in ['add','mul']
    assert attention_method in ['att','avg']

    with tf.variable_scope(scope,"GCModule"):

        if attention_method == 'avg':
            context = global_avg_pool2D(x)#[N,1,1,C]
        else:
            n,h,w,c=x.get_shape().as_list()
            context_mask = conv(x,1,1)# [N, H, W,1]
            context_mask = tf.reshape(context_mask,shape=tf.convert_to_tensor([tf.shape(x)[0], -1, 1]))# [N, H*W, 1]
            context_mask=tf.transpose(context_mask,perm=[0,2,1])# [N, 1, H*W]
            context_mask = tf.nn.softmax(context_mask,axis=2)# [N, 1, H*W]

            input_x = tf.reshape(x, shape=tf.convert_to_tensor([tf.shape(x)[0], -1,c]))# [N,H*W,C]

            context=tf.matmul(context_mask,input_x)# [N, 1, H*W] x [N,H*W,C] =[N,1,C]
            context=tf.expand_dims(context,axis=1)#[N,1,1,C]

        context=conv(context,squeeze_depth,1)
        context=slim.layer_norm(context)
        context=tf.nn.relu(context)
        context=conv(context,c,1)#[N,1,1,C]

        if fuse_method=='mul':
            context=tf.nn.sigmoid(context)
            out=context*x
        else:
            out=context+x

        return out


if __name__=='__main__':

    inputs=tf.placeholder(tf.float32,shape=[None,64,64,128])
    input_array=np.ones((1,64,64,128),dtype=np.float32)

    out=global_context_module(inputs,squeeze_depth=16)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        output=sess.run([out],feed_dict={inputs:input_array})
        print(output[0].shape)