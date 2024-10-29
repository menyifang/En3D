# Copyright (c) Alibaba, Inc. and its affiliates.
# coding=utf-8
import tensorflow as tf
import numpy as np
import cv2
import os

if tf.__version__ >= '2.0':
    print("tf version >= 2.0")
    tf = tf.compat.v1
    tf.disable_eager_execution()


class human_segmenter(object):
    def __init__(self, model_path,is_encrypted_model=False):
        super(human_segmenter, self).__init__()
        f = tf.gfile.FastGFile(model_path, 'rb')
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        persisted_graph = tf.import_graph_def(graph_def, name='')

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        self.sess = tf.InteractiveSession(graph=persisted_graph, config=config)

        print("human_segmenter init done")
    
    def image_preprocess(self, img):
        if len(img.shape) == 2:
            img = np.dstack((img, img, img))
        elif img.shape[2] == 4:
            img = img[:, :, :3]
        img = img[:, :, ::-1]
        img = img.astype(np.float)
        return img
    
    def run(self, img):
        image_feed = self.image_preprocess(img)
        output_img_value, logits_value = self.sess.run([self.sess.graph.get_tensor_by_name("output_png:0"), self.sess.graph.get_tensor_by_name("if_person:0")],
                                                  feed_dict={self.sess.graph.get_tensor_by_name("input_image:0"): image_feed})
        # mask = output_img_value[:,:,-1]
        output_img_value = cv2.cvtColor(output_img_value, cv2.COLOR_RGBA2BGRA)
        return output_img_value

    def run_head(self, img):
        image_feed = self.image_preprocess(img)
        # image_feed = image_feed/255.0
        output_alpha = self.sess.run(self.sess.graph.get_tensor_by_name('output_alpha:0'),
                                     feed_dict={'input_image:0': image_feed})

        return output_alpha
    
    def get_human_bbox(self, mask):
        '''
        
        :param mask:
        :return: [x,y,w,h]
        '''
        print('dtype:{}, max:{},shape:{}'.format(mask.dtype, np.max(mask), mask.shape))
        ret, thresh = cv2.threshold(mask,127,255,0)
        contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return None
        
        contoursArea = [cv2.contourArea(c) for c in contours]
        max_area_index = contoursArea.index(max(contoursArea))
        bbox = cv2.boundingRect(contours[max_area_index])
        return bbox
    
    
    def release(self):
        self.sess.close()




