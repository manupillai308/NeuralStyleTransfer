#!/usr/bin/env python
# coding: utf-8

# In[22]:


tf.reset_default_graph()


# In[23]:


import tensorflow as tf
from tensorflow.python.platform import gfile

GRAPH_PB_PATH = './alexnet_frozen.pb'
with tf.Session() as sess:
    print("load graph")
    with gfile.FastGFile(GRAPH_PB_PATH,'rb') as f:
        graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
    # sess.graph.as_default()
    tf.import_graph_def(graph_def, name='') # importing into the main graph


# In[24]:


graph = tf.get_default_graph()
inp = graph.get_tensor_by_name('Placeholder:0')
layer = graph.get_tensor_by_name('conv5_1:0')


# In[65]:


import cv2
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np

content_image = cv2.imread('./content.JPG')
style_image = cv2.imread('./style.jpg')

content_im = cv2.resize(content_image, (227,227))
style_im = cv2.resize(style_image, (227,227))
content_im, style_im = np.expand_dims(content_im[...,[2,1,0]],axis=0), np.expand_dims(style_im[...,[2,1,0]], axis=0)

plt.imshow(content_im[0])
plt.show()
plt.imshow(style_im[0])
plt.show()


# In[81]:


layers_with_weights = [('conv1_1:0', 0.1), ('conv2_1:0', 0.1), ('conv3_1:0', 0.3), ('conv4_1:0', 0.3), ('conv5_1:0', 0.3)]


# In[27]:


def content_loss(content, generated):
    b, nh, nw, nc = generated.get_shape().as_list()
    loss = 1 * tf.reduce_sum(tf.square(content-generated))/(4 * nh * nw * nc)
    return loss

def gram_matrix(mat):
    gram_mat = tf.matmul(mat, mat, transpose_b= True)
    return gram_mat

def style_loss(style, generated):
    b, nh, nw, nc = generated.get_shape().as_list()
    style_reshape = tf.reshape(tf.transpose(style, perm=[0,3,1,2]), [-1,nc, nh*nw])
    gen_reshape = tf.reshape(tf.transpose(generated, perm=[0,3,1,2]), [-1,nc, nh*nw])
    gram_gen = gram_matrix(gen_reshape)
    gram_sty = gram_matrix(style_reshape)
    loss = 1 * tf.reduce_sum(tf.square(gram_gen-gram_sty))/(4 * nh * nw * nc * nh * nw * nc)
    return loss

def style_loss_with_layers(feed_dict, layers_and_weights):
    cum_loss = 0
    for layer_name, weight in layers_and_weights:
        tensor = graph.get_tensor_by_name(layer_name)
        output = sess.run(tensor, feed_dict=feed_dict) # feed_dict => style as input
        sty_loss = style_loss(output, tensor)
        cum_loss+=(weight*sty_loss)
    return cum_loss

def generated_random_im(content_im, noise_ratio=0.5):
    gen_im = tf.Variable(noise_ratio * tf.random_uniform(dtype=tf.float32, shape=[1,227,227,3], minval=0, maxval=255) + (1- noise_ratio) * content_im)
    return gen_im


# In[82]:


generated_im = generated_random_im(content_im=content_im, noise_ratio=0.4)


# In[84]:


sess = tf.InteractiveSession()


# In[85]:


alpha = 15
beta = 40


# In[86]:


total_loss = alpha * content_loss(layer.eval(feed_dict={inp:content_im}), layer) + beta * style_loss_with_layers(feed_dict={inp:style_im},layers_and_weights=layers_with_weights)


# In[87]:


optimizer = tf.train.AdamOptimizer(learning_rate = 2.0, beta1=0.9, beta2=0.999)


# In[88]:


gradients = optimizer.compute_gradients(total_loss, [inp])[0][0]


# In[89]:


init = tf.global_variables_initializer()
init.run()


# In[ ]:


iteration = 400
gen_im = generated_im.eval()
for i in range(iteration):
    gradients_val = sess.run(gradients, feed_dict={inp:gen_im})
    apply_grad = optimizer.apply_gradients([(gradients_val, generated_im)])
    if i == 0:
        tf.variables_initializer(optimizer.variables()).run()
    apply_grad.run(feed_dict={inp:gen_im})
    gen_im = generated_im.eval()
    if i%20 == 0:
        print('\rTotal Current Loss: ',total_loss.eval(feed_dict={inp:gen_im}), end='')
        cv2.imwrite('./out/%s.jpg' % str(i), gen_im[0][..., [2,1,0]])
cv2.imwrite('./out/%s.jpg' % str(i), gen_im[0][..., [2,1,0]]) #saving last image


# In[91]:


sess.close()

