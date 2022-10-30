import numpy as np
import tensorflow as tf
print("Loading model...")

model = tf.saved_model.load('model_temp')
model.trainable = False

def make_grid(h, w):
    wv, hv = tf.meshgrid(tf.range(h), tf.range(w))
    grid = tf.stack((wv, hv), 2)
    grid = tf.concat([grid, grid, grid], axis=2)
    grid = tf.reshape(grid, [h, w, 3, -1])
    return grid

class MLP(tf.Module):
    def __init__(self, name=None): 
        super(MLP, self).__init__(name=name)
        self.model = model

    @tf.function(input_signature=[tf.TensorSpec(shape=[352,352, 3], dtype=tf.float32)])
    def __call__(self, x):
        x = tf.expand_dims(x, axis=0)
        x = tf.transpose(x, perm=[0, 3, 1, 2])
        x = tf.divide(x, 255.0)
        x = self.model(**{'input.1': x})
        preds = [x['777'], x['778'], x['779'],x['780'], x['781'], x['782']]
        #加载anchor配置
        # anchors = np.array(cfg["anchors"])
        anchors = np.array([12.64, 19.39, 37.88, 51.48, 55.71, 138.31, 126.91, 78.23, 131.57, 214.55, 279.92, 258.87]) 
        # anchors = torch.from_numpy(anchors.reshape(len(preds) // 3, 3, 2)).to(device) , 3 = 3
        anchors = anchors.reshape(len(preds) // 3, 3, 2)
        anchors = tf.convert_to_tensor(anchors)
        
        # output_bboxes = []

        # for index in range(len(preds) // 3):
        # bacth_bboxes = []
        index = 0
        reg_preds = preds[index * 3]
        obj_preds = preds[(index * 3) + 1]
        cls_preds = preds[(index * 3) + 2]

        
        #for r, o, c in zip(reg_preds, obj_preds, cls_preds):
        for i in range(tf.shape(reg_preds)[0]):

            r, o, c = reg_preds[i], obj_preds[i], cls_preds[i]
            
            # r = r.permute(1, 2, 0)
            r = tf.transpose(r, perm=[1, 2, 0])
            
            # r = r.reshape(r.shape[0], r.shape[1], 3, -1)
            r = tf.reshape(r, [tf.shape(r)[0], tf.shape(r)[1], 3, -1])

            # o = o.permute(1, 2, 0)
            o = tf.transpose(o, perm=[1, 2, 0])
            # o = o.reshape(o.shape[0], o.shape[1], 3, -1)
            o = tf.reshape(o, [tf.shape(o)[0], tf.shape(o)[1], 3, -1])

            # c = c.permute(1, 2, 0)
            c = tf.transpose(c, perm=[1, 2, 0])
            # c = c.reshape(c.shape[0],c.shape[1], 1, c.shape[2])
            c = tf.reshape(c, [tf.shape(c)[0], tf.shape(c)[1], 1, tf.shape(c)[2]])
            # c = c.repeat(1, 1, 3, 1)
            c = tf.repeat(c, [3],axis=2)

            # anchor_boxes = tf.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1)

            #计算anchor box的cx, cy
            # grid = make_grid(r.shape[0], r.shape[1])
            grid = tf.cast(make_grid(tf.shape(r)[0], tf.shape(r)[1]), tf.float32)
            # stride = 352 /  r.shape[0]
            stride = tf.cast(352 /  tf.shape(r)[0], tf.float32)
            
            # anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride
            anchor_boxes_0_2 = ((tf.sigmoid(r[:, :, :, :2]) * 2.0 - 0.5) + grid) * stride
            

            #计算anchor box的w, h
            anchors_cfg = tf.cast(anchors[index], tf.float32)
            
            # anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg # wh
            anchor_boxes_2_4 = (tf.sigmoid(r[:, :, :, 2:4]) * 2) ** 2 * anchors_cfg # wh
            
            #计算obj分数
            # anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()
            anchor_boxes_4 = tf.sigmoid(o[:, :, :, 0:1])
            #计算cls分数
            # anchor_boxes[:, :, :, 5:] = F.softmax(c[:, :, :, :], dim = 3)
            anchor_boxes_5_ = tf.nn.softmax(c[:, :, :, :], axis = 3)

            #torch tensor 转为 numpy array
            # anchor_boxes = anchor_boxes.cpu().detach().numpy() 
            anchor_boxes = tf.concat([anchor_boxes_0_2, anchor_boxes_2_4, anchor_boxes_4, anchor_boxes_5_], axis=3)

            # bacth_bboxes.append(anchor_boxes) 
            batch_bboxes = [anchor_boxes]

            #n, anchor num, h, w, box => n, (anchor num*h*w), box
            # bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
            batch_bboxes = tf.convert_to_tensor(batch_bboxes)
            # bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1]) 
            batch_bboxes = tf.reshape(batch_bboxes, [tf.shape(batch_bboxes)[0], -1, tf.shape(batch_bboxes)[-1]])

        index = 1
        reg_preds = preds[index * 3]
        obj_preds = preds[(index * 3) + 1]
        cls_preds = preds[(index * 3) + 2]

        
        #for r, o, c in zip(reg_preds, obj_preds, cls_preds):
        for i in range(tf.shape(reg_preds)[0]):

            r, o, c = reg_preds[i], obj_preds[i], cls_preds[i]
            
            # r = r.permute(1, 2, 0)
            r = tf.transpose(r, perm=[1, 2, 0])
            
            # r = r.reshape(r.shape[0], r.shape[1], 3, -1)
            r = tf.reshape(r, [tf.shape(r)[0], tf.shape(r)[1], 3, -1])

            # o = o.permute(1, 2, 0)
            o = tf.transpose(o, perm=[1, 2, 0])
            # o = o.reshape(o.shape[0], o.shape[1], 3, -1)
            o = tf.reshape(o, [tf.shape(o)[0], tf.shape(o)[1], 3, -1])

            # c = c.permute(1, 2, 0)
            c = tf.transpose(c, perm=[1, 2, 0])
            # c = c.reshape(c.shape[0],c.shape[1], 1, c.shape[2])
            c = tf.reshape(c, [tf.shape(c)[0], tf.shape(c)[1], 1, tf.shape(c)[2]])
            # c = c.repeat(1, 1, 3, 1)
            c = tf.repeat(c, [3],axis=2)

            # anchor_boxes = tf.zeros(r.shape[0], r.shape[1], r.shape[2], r.shape[3] + c.shape[3] + 1)

            #计算anchor box的cx, cy
            # grid = make_grid(r.shape[0], r.shape[1])
            grid = tf.cast(make_grid(tf.shape(r)[0], tf.shape(r)[1]), tf.float32)
            # stride = 352 /  r.shape[0]
            stride = tf.cast(352 /  tf.shape(r)[0], tf.float32)
            
            # anchor_boxes[:, :, :, :2] = ((r[:, :, :, :2].sigmoid() * 2. - 0.5) + grid) * stride
            anchor_boxes_0_2 = ((tf.sigmoid(r[:, :, :, :2]) * 2.0 - 0.5) + grid) * stride
            

            #计算anchor box的w, h
            anchors_cfg = tf.cast(anchors[index], tf.float32)
            
            # anchor_boxes[:, :, :, 2:4] = (r[:, :, :, 2:4].sigmoid() * 2) ** 2 * anchors_cfg # wh
            anchor_boxes_2_4 = (tf.sigmoid(r[:, :, :, 2:4]) * 2) ** 2 * anchors_cfg # wh
            
            #计算obj分数
            # anchor_boxes[:, :, :, 4] = o[:, :, :, 0].sigmoid()
            anchor_boxes_4 = tf.sigmoid(o[:, :, :, 0:1])
            #计算cls分数
            # anchor_boxes[:, :, :, 5:] = F.softmax(c[:, :, :, :], dim = 3)
            anchor_boxes_5_ = tf.nn.softmax(c[:, :, :, :], axis = 3)

            #torch tensor 转为 numpy array
            # anchor_boxes = anchor_boxes.cpu().detach().numpy() 
            anchor_boxes = tf.concat([anchor_boxes_0_2, anchor_boxes_2_4, anchor_boxes_4, anchor_boxes_5_], axis=3)

            # bacth_bboxes.append(anchor_boxes) 
            batch_bboxes1 = [anchor_boxes]

            #n, anchor num, h, w, box => n, (anchor num*h*w), box
            # bacth_bboxes = torch.from_numpy(np.array(bacth_bboxes))
            batch_bboxes1 = tf.convert_to_tensor(batch_bboxes1)
            # bacth_bboxes = bacth_bboxes.view(bacth_bboxes.shape[0], -1, bacth_bboxes.shape[-1]) 
            batch_bboxes1 = tf.reshape(batch_bboxes1, [tf.shape(batch_bboxes1)[0], -1, tf.shape(batch_bboxes1)[-1]])

        # merge
        output = tf.concat([batch_bboxes, batch_bboxes1], 1)
        
        # non-maximum suppression
        prediction = output
        conf_thres = 0.1
        iou_thres = 0.4
        classes=None
        
        
        nc = prediction.shape[2] - 5  # number of classes

        # Settings
        # (pixels) minimum and maximum box width and height
        max_wh = 4096
        max_det = 300  # maximum number of detections per image
        max_nms = 30000  # maximum number of boxes into torchvision.ops.nms()
        time_limit = 1.0  # seconds to quit after
        multi_label = nc > 1  # multiple labels per box (adds 0.5ms/img)


        xi = 0
        x = prediction[0]

        scores = x[:, 5] * x[:,4]
        # Box (center x, center y, width, height) to (x1, y1, x2, y2)
        bbox = x[:, :4]
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right

        bbox_0 = bbox[:, 0] - bbox[:, 2] / 2  # top left x
        bbox_1 = bbox[:, 1] - bbox[:, 3] / 2  # top left y
        bbox_2 = bbox[:, 0] + bbox[:, 2] / 2  # bottom right x
        bbox_3 = bbox[:, 1] + bbox[:, 3] / 2  # bottom right y
        
        bbox_xyxy = tf.stack([bbox_0, bbox_1, bbox_2, bbox_3], axis=1)

        # c = 0 * max_wh  # classes
        # # boxes (offset by class), scores
        # bbox_xyxy = bbox_xyxy + c

        # perform nms
        selected_indices = tf.image.non_max_suppression(bbox_xyxy,
                                                        scores,
                                                        200,
                                                        iou_threshold=iou_thres,
                                                        score_threshold=conf_thres,
                                                        name='non_max_suppression')
        output = tf.concat([bbox_xyxy, scores[...,tf.newaxis]], axis=1, name='output')
        output = tf.gather(output, selected_indices) 
            
        return output
    
dummy_input = np.ones((352, 352, 3),dtype=np.float32)
test = MLP()
out = test(dummy_input)
for o in out:
    print(o.shape)

tf.saved_model.save(test, 'model_temp_2')
print("Converting model...")
converter = tf.lite.TFLiteConverter.from_saved_model('model_temp_2')
tflite_model = converter.convert()

with open('yolo-fastest-v2.tflite', 'wb') as f:
    f.write(tflite_model)

print("Convert successfully!")
