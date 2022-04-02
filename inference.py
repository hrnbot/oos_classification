import time
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from PIL import Image
import cv2
import glob
import os
import shutil

def clip_coords(boxes, shape):
    # Clip bounding xyxy bounding boxes to image shape (height, width)
    if isinstance(boxes, torch.Tensor):  # faster individually
        boxes[:, 0].clamp_(0, shape[1])  # x1
        boxes[:, 1].clamp_(0, shape[0])  # y1
        boxes[:, 2].clamp_(0, shape[1])  # x2
        boxes[:, 3].clamp_(0, shape[0])  # y2
    else:  # np.array (faster grouped)
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, shape[1])  # x1, x2
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, shape[0])  # y1, y2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def xywh2xyxys(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y =  np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def xywh2xyxy(x):
    # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
    y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
    y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
    y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
    return y

def nms_boxes_func(boxes, box_confidences, nms_threshold):
    """Apply the Non-Maximum Suppression (NMS) algorithm on the bounding boxes with their
    confidence scores and return an array with the indexes of the bounding boxes we want to
    keep (and display later).
    Keyword arguments:
    boxes -- a NumPy array containing N bounding-box coordinates that survived filtering,
    with shape (N,4); 4 for x,y,height,width coordinates of the boxes
    box_confidences -- a Numpy array containing the corresponding confidences with shape N
    """

    x_coord = boxes[:, 0]
    y_coord = boxes[:, 1]
    width = boxes[:, 2]
    height = boxes[:, 3]
    areas = width * height

    # Sorting bounding boxes confidence wise
    ordered = box_confidences.argsort()[::-1]

    keep = list()
    #Check for NMS
    while ordered.size > 0:
        # Index of the current element:
        i = ordered[0]
        keep.append(i)
        xx1 = np.maximum(x_coord[i], x_coord[ordered[1:]])
        yy1 = np.maximum(y_coord[i], y_coord[ordered[1:]])
        xx2 = np.minimum(x_coord[i] + width[i], x_coord[ordered[1:]] + width[ordered[1:]])
        yy2 = np.minimum(y_coord[i] + height[i], y_coord[ordered[1:]] + height[ordered[1:]])

        width1 = np.maximum(0.0, xx2 - xx1 + 1)
        height1 = np.maximum(0.0, yy2 - yy1 + 1)
        intersection = width1 * height1
        union = (areas[i] + areas[ordered[1:]] - intersection)

        # Compute the Intersection over Union (IoU) score:
        iou = intersection / union

        # The goal of the NMS algorithm is to reduce the number of adjacent bounding-box
        # candidates to a minimum. In this step, we keep only those elements whose overlap
        # with the current bounding box is lower than the threshold:
        indexes = np.where(iou <= nms_threshold)[0]
        ordered = ordered[indexes + 1]
    keep = np.array(keep)
    return keep

def postprocess(bbs, class_ids, scores, nms_threshold, wh_format=True):
        """
        Postprocesses the inference output
        Args:
            outputs (list of float): inference output
            min_confidence (float): min confidence to accept detection
            analysis_classes (list of int): indices of the classes to consider

        Returns: list of list tuple: each element is a two list tuple (x, y) representing the corners of a bb
        """
        nms_boxes, nms_categories, nscores = list(), list(), list()
        box=bbs
        confidence=scores
        category=class_ids
        keep = nms_boxes_func(box, confidence, nms_threshold)
        nms_boxes.append(box[keep])
        nms_categories.append(category[keep])
        nscores.append(confidence[keep])
        #  
        if len(nms_boxes) == 0:
            return [], [], []

        return nms_boxes, nms_categories, nscores


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.8, classes=None, agnostic=False, multi_label=False,
                        labels=(), max_det=300):
    """Runs Non-Maximum Suppression (NMS) on inference results

    Returns:
         list of detections, on (n,6) tensor per image [xyxy, conf, cls]
    """

    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates
    # print(conf_thres)
    #  
    # Checks
    assert 0 <= conf_thres <= 1, f'Invalid Confidence threshold {conf_thres}, valid values are between 0.0 and 1.0'
    assert 0 <= iou_thres <= 1, f'Invalid IoU {iou_thres}, valid values are between 0.0 and 1.0'

    # Settings
    min_wh, max_wh = 2, 7680 # (pixels) minimum and maximum box width and height
    max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)
    merge = False  # use merge-NMS

    t = time.time()
    # output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
    output=np.zeros((0,6))
    # print(prediction)
    # print(prediction.shape)
     
    for xi, x in enumerate(prediction):  # image index, image inference
        # Apply constraints
        # print(x[..., 2:4] < min_wh)
        x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
        # print(x.shape,xc.shape,xc[xi])
        #  
        # print(x)
        x = x[xc[xi]]  # confidence
        # print(xc[xi].tolist())
         
        if not x.shape[0]:
            continue

        # Compute conf
        x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        box = xywh2xyxy(x[:, :4])
        # print(box.shape)
         
        labels=np.argmax(x[:,5:],axis=1)
        conf=x[:,4]

        return box,labels,conf

    return None, None, None

class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        """
        Class to device memory into Host and Device
        """
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TRTLoader:
    def __init__(self, trt_engine_path, model_w, model_h, labels_text, threshold, nms_threshold):
        """
        Create Model architecture for engine inference
        Args:
            trt_engine_path (string): path of engine generated using tao converter
            model_w (int): width of image given in model architecture
            model_h (int): height of image given in model architecture
            labels_text (list(str)): a list of labels
            threshold (float): threshold for detection acceptance
            nms_threshold (float): threshold for nms acceptance
        """
        # Assign all the function variables to class variables
        self.model_w = model_w
        self.model_h = model_h
        self.threshold = threshold
        self.labels_text = labels_text

        # Load TRT Engine and allocate Buffer memory
        self.trt_engine = self.load_engine(trt_engine_path)
        self.context = self.trt_engine.create_execution_context()
        inputs, outputs, bindings, stream = self.allocate_buffers(self.trt_engine)
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings
        self.stream = stream
        self.nms_threshold = nms_threshold

    def load_engine(self, engine_path):
        """
        Loads TRT Engine in runtime memory with CUDA cores
        Args:
            engine_path (string): path of trt converted model

        Returns:
            [engine object]: runtime loaded memory
        """
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(None, '')
        # Create trt engine
        trt_runtime = trt.Runtime(TRT_LOGGER)

        # Read engine path as binary
        with open(engine_path, "rb") as f:
            engine_data = f.read()

        # Deserialize engine architecture object and weight for runtime inference
        engine = trt_runtime.deserialize_cuda_engine(engine_data)
        return engine

    def allocate_buffers(self, engine, batch_size=1):
        """Allocates host and device buffer for TRT engine inference.
        This function is Similar to the one in common.py, but
        converts network outputs (which are np.float32) appropriately
        before writing them to Python buffer. This is needed, since
        TensorRT plugins doesn't support output type description, and
        in our particular case, we use NMS plugin as network output.
        Args:
            engine (trt.ICudaEngine): TensorRT engine
        Returns:
            inputs [HostDeviceMem]: engine input memory
            outputs [HostDeviceMem]: engine output memory
            bindings [int]: buffer to device bindings
            stream (cuda.Stream): CUDA stream for engine inference synchronization
        """
        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        # Current NMS implementation in TRT only supports DataType.FLOAT but
        # it may change in the future, which could brake this sample here
        # when using lower precision [e.g. NMS output would not be np.float32
        # anymore, even though this is assumed in binding_to_type]
        # binding_to_type = {
        #     'Input': np.float32,
        #     'BatchedNMS': np.int32,
        #     'BatchedNMS_1': np.float32,
        #     'BatchedNMS_2': np.float32,
        #     'BatchedNMS_3': np.float32
        # }

        binding_to_type = {
            'input_1': np.float32,
            'predictions/Softmax': np.float32
            # '350': np.float32,
            # # '416': np.float32,
            # # '482': np.float32
            # '339': np.int32,
            # '391': np.float32,
            # '443': np.float32
            
        }

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(binding)) * batch_size
            dtype = binding_to_type[str(binding)]
            # Allocate host and device buffers
            #size = abs(size)
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))

        return inputs, outputs, bindings, stream

    def letterbox(self,im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
        # Resize and pad image while meeting stride-multiple constraints

        shape = im.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        # print(new_shape[0] , shape[0], new_shape[1] , shape[1])
        #  
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        ratio = r, r  # width, height ratios
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
        if auto:  # minimum rectangle
            dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
        elif scaleFill:  # stretch
            dw, dh = 0.0, 0.0
            new_unpad = (new_shape[1], new_shape[0])
            ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

        dw /= 2  # divide padding into 2 sides
        dh /= 2

        if shape[::-1] != new_unpad:  # resize
            im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
        return im, ratio, (dw, dh)


    def preprocess_image(self, img):
        """Pre processing and transforming image as per model input

        Args:
            arr (numpy): numpy image array of 3 dims

        Returns:
            [numpy]: numpy image array of 3 dims
        """
        # Pre process image
        
        # img = img.resize((640,256))
        img=cv2.resize(img,(self.model_w,self.model_h))
        # img = self.letterbox(img, [self.model_h,self.model_w], stride=64, auto=False)[0]
        # print(img)
        #  
        # img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        # print(img)
        #  
        img = np.array(img, dtype=np.float)
        # print(img.shape)
        # img = img.transpose((2, 1, 0))
        # print(img.shape)
        # exit()
        #  
        # print(img)
        #  
        img /= 255

        # HWC -> CHW

        # Normalize to [0.0, 1.0] interval (expected by model)
        # print(img,img.shape)
        # exit()
        img = img.ravel()
        # print(img)
        # print(img.shape)
        #  
        # exit()
        return img


    def do_inference(self, context, bindings, inputs, outputs, stream, batch_size=1):
        # This function is generalized for multiple inputs/outputs.
        # inputs and outputs are expected to be lists of HostDeviceMem objects.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        print(outputs)
        #  
        # Run inference.
        context.execute_async(
            batch_size=batch_size, bindings=bindings, stream_handle=stream.handle
        )
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream

        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def predict_inference(self, image):
        """Infers model on batch of same sized images resized to fit the model.
        Args:
            image_paths (str): paths to images, that will be packed into batch
                and fed into model
        """
        img = self.preprocess_image(image)
        # img=cv2.resize(image,(self.model_w,self.model_h))
        print(img,img.shape)
        #  
        # print(img)
        # exit()
        inputs = self.inputs
        outputs = self.outputs
        bindings = self.bindings
        stream = self.stream
        np.copyto(inputs[0].host, img)

        # When infering on single image, we measure inference
        # time to output it to the user

        # Fetch output from the model
        # print(inputs)
        # print(inputs[0].host.shape)
        #  
        print(inputs)
        detection_out = self.do_inference(
            self.context, bindings=bindings, inputs=inputs, outputs=outputs, stream=stream
        )
        print(detection_out)
        
        #  
        # Output inference time
        """print(
            "TensorRT inference time: {} ms".format(
                int(round((time.time() - inference_start_time) * 1000))
            )
        )"""
        # perform non-max suppression
        # box, labels, conf = non_max_suppression(
        #     detection_out[3].reshape(1, 10080, 7),       #TODO: How did we get this number?
        #     conf_thres=self.threshold,
        #     iou_thres=self.nms_threshold,
        #     classes=list(labels_text)
        #     )
        # # No predictions found
        # if box is None:
        #     return []

        # # perform post-processing.
        # bounding_boxes, labels, confidences = postprocess(
        #     bbs=box,
        #     scores=conf,
        #     class_ids=labels,
        #     nms_threshold=self.nms_threshold)

        # put into format cloud can understand + scale-up predictions to original image size.
        # predictions = []
        # x_scale = image.shape[1] / self.model_w
        # y_scale = image.shape[0] / self.model_h
        # for bounding_box, label, confidence in zip(bounding_boxes[0], labels[0], confidences[0]):
        #     pred = {}
        #     bounding_box = list(bounding_box)
        #     pred['label'] = self.labels_text[int(label)]
        #     pred['confidence'] = round(confidence.item(), 2)
        #     pred['label_index']=int(label)
            # pred['boundingBox'] = {
            #     'left': int(bounding_box[0] * x_scale),
            #     'top': int(bounding_box[1] * y_scale),
            #     'right': int(bounding_box[2] * x_scale),
            #     'bottom': int(bounding_box[3] * y_scale),
            #     'width': int((bounding_box[2] - bounding_box[0]) * x_scale),
            #     'height': int((bounding_box[3] - bounding_box[1]) * y_scale)
            # }
            # predictions.append(pred)

        return "predictions"


if __name__ == '__main__':
    # path of engine file converted from etlt
    # engine_path = "shelf-object-detection.engine"
    engine_path = "final_modelv1.trt"

    # input and output image for Inference
    images_input_path = "input"

    if os.path.exists('output/'):
        shutil.rmtree('output/')

    os.makedirs('output/', exist_ok=True)

    # labels array of output
    labels_text=["bottle", "void"]
    length=len(labels_text)
    colors =[list(np.random.random(size=3) * 256)for i in range(length)]
    # labels_text=['bottle', 'void', 'can', 'carton', 'container', 'eggs', 'pouch', 'v', 'void']

    # threshold of labels to accept
    threshold=0.25

    # Non maximum suppression threshold to accept
    nms_threshold=0.5

    # Load Image in Runtime for Inference
    engine = TRTLoader(trt_engine_path=engine_path, model_w=224, model_h=224,
                       labels_text=labels_text, threshold=threshold, nms_threshold=nms_threshold)

    # Get all image paths
    path="val/oos/"
    images=glob.glob(path+"*.jpg")
    all_start = time.time()
    # Inference over all the images
    print("[systime] [inference_time] [num_predictions]: output_file")
    for image in images:
        start_time = time.time()

        # Open Image in pillow
        # img = Image.open(image)
        img=cv2.imread(image)
        
        predictions = engine.predict_inference(img)
        print("[{:.2f}s] [{:.2f}s] [{:3}]: output/{}".format(
            time.time() - all_start,
            time.time() - start_time,
            len(predictions),
            os.path.basename(image)))

        # Annotate results
        img = cv2.imread(image)
        for pred in predictions:
        
            img = cv2.rectangle(
                img,
                (pred['boundingBox']['left'], pred['boundingBox']['top']),
                (pred['boundingBox']['right'], pred['boundingBox']['bottom']),
                colors[pred['label_index']],
                2)

            img = cv2.putText(
                img,
                pred['label'] + " " + str(pred['confidence']),
                (pred['boundingBox']['left'], pred['boundingBox']['top']),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                colors[pred['label_index']],
                2)

        cv2.imwrite("output/"+os.path.basename(image), img)