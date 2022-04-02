import os
import time

import cv2
#import matplotlib.pyplot as plt
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image



class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def load_engine(trt_runtime, engine_path):
    with open(engine_path, "rb") as f:
        engine_data = f.read()
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    return engine


def allocate_buffers(engine):
    # Determine dimensions and create page-locked memory buffers (i.e. won't be swapped to disk) to hold host inputs/outputs.
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=trt.nptype(trt.float32))
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=trt.nptype(trt.float32))
    # Allocate device memory for inputs and outputs.
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    # Create a stream in which to copy inputs/outputs and run inference.
    stream = cuda.Stream()
    return h_input, d_input, h_output, d_output, stream

def load_normalized_test_case(test_image, pagelocked_buffer):
    # Converts the input image to a CHW Numpy array
    def normalize_image(image):
        # Resize, antialias and transpose the image to CHW.
        c, h, w = 3, model_w, model_h
        return np.asarray(image.resize((w, h), Image.ANTIALIAS)).transpose([2, 0, 1]).astype(trt.nptype(trt.float32)).ravel()

    # Normalize the image and copy to pagelocked memory.
    np.copyto(pagelocked_buffer, normalize_image(Image.open(test_image)))
    return test_image

def do_inference(context, h_input, d_input, h_output, d_output, stream):
    # Transfer input data to the GPU.
    cuda.memcpy_htod_async(d_input, h_input, stream)
    # Run inference.
    context.execute_async(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    # Synchronize the stream
    stream.synchronize()
    return h_output,h_input


if __name__ == '__main__':
    
    neg = 0
    pos = 0
    count = 0
    
    # TensorRT logger singleton
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    trt_engine_path = os.path.join("final_modelv1.trt")
    
    trt_runtime = trt.Runtime(TRT_LOGGER)
    trt_engine = load_engine(trt_runtime, trt_engine_path)
    
    # This allocates memory for network inputs/outputs on both CPU and GPU
    h_input, d_input, h_output, d_output, stream = allocate_buffers(trt_engine)
    
    # Execution context is needed for inference
    context = trt_engine.create_execution_context()


    # -------------- MODEL PARAMETERS FOR THE MODEL --------------------------------
    model_h = 224
    model_w = 224
    img_dir = "val/"
    
    folders = os.listdir(img_dir)
    
    for sub_folder in folders:
        #loop over the folders
        images = os.listdir(img_dir + sub_folder)
        
        for i in images:
            #loop over the images
            
            test_image = img_dir + sub_folder + "/" + i
            
            # labels_file = "labels.txt"
            # labels = open(labels_file, 'r').read().split('\n')
            labels=["not-oos","oos"]
            test_case = load_normalized_test_case(test_image, h_input)
            
            start_time = time.time()
            h_output,h_input = do_inference(context, h_input, d_input, h_output, d_output, stream)
            pred = labels[np.argmax(h_output)]
            
            #print (test_image)
            print (sub_folder,"class: ",pred,", Confidence: ", max(h_output))
            print ("Inference Time : ",time.time()-start_time)
            
            if sub_folder == pred:
                 pos+=1
            else:
                neg+=1
            # if sub_folder == "positive":
            #     pos+=1
            #     print (test_image)
                #time.sleep(3)
                
            #time.sleep(1)
            count += 1
    
    print ("Total Number of items in the directory : ",count)
    print ("Total number of Positive Classified Items : ",pos)
    print ("Total number of Negative Classified Items : ",neg)
    
    
