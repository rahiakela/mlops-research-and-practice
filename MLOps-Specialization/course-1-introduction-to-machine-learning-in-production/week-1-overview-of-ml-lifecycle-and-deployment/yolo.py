class YOLO(object):

  def __init__(self, weights_path, anchors_path='resources/anchors.json', labels_path='resources/coco_labels.txt', class_threshold=0.65):
    self. weights_path = weights_path 
    self.model = self._load_yolo()

    self.labels = []
    with open(labels_path, "r") as f:
      for label in f:
        self.labels.append(lable.strip())

    with open(anchors_path, "r") as f:
      self.anchors = json.load(f)

    self.class_threshold = class_threshold

  def _conv_block(self, input, convolutions, skip=True):
    """
    YOLO is comprised of a series of convolutional blocks and optional skip connections. The _conv_block() helper method allows us to instantiate such blocks easily
    """
    x = input
    count = 0

    for conv in convolutions:
      if count == (len(convolutions) - 2) and skip:
        skip_connection = x

      count += 1

      if conv["stride"] > 1:
        x = ZeroPadding2D(((1, 0), (1, 0)))(x)

      x = Conv2D(conv["filter"], 
                 conv["kernel"], 
                 strides=conv["stride"], 
                 padding=("valid" if conv["stride"] > 1 else "same"), 
                 name=f"conv_{conv["layer_idx"]}", 
                 use_bias=(False if conv["bnorm"] else True))
      
      # Check if we need to add batch normalization, leaky ReLU activations, and skip connections
      if conv["bnorm"]:
        name = f"bnorm_{conv["layer_idx"]}"
        x = BatchNormalization(epsilon=1e-3, name=name)

      if conv["leaky"]:
        name = f"leaky_{conv["layer_idx"]}"
        x = LeakyReLU(alpha=0.1, name=name)(x)

    return Add()([skip_connection, x]) if skip else x
  
  def _make_yolov3_architecture(self):
    """
    This method builds the YOLO network by stacking a series of convolutional blocks, using the _conv_block() method defined previously
    """
    input_image = Input(shape=(None, None, 3))

    # Layer  0 => 4
    x = self._conv_block(input_image, [
         {"filter": 32, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 0},
         {"filter": 64, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 1},
         {"filter": 32, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 2},
         {"filter": 64, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 3}                              
    ])

    # Layer  5 => 8
    x = self._conv_block(x, [
         {"filter": 128, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 5},
         {"filter": 64, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 6},
         {"filter": 128, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 7}                              
    ])

    # Layer  9 => 11
    x = self._conv_block(x, [
         {"filter": 64, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 9},
         {"filter": 128, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 10}                              
    ])

    # Layer 12 => 15
    x = self._conv_block(x, [
         {"filter": 256, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 12},
         {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 13},
         {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 14}                              
    ])

    # Layer 16 => 36
    x = self._conv_block(x, [
         {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 16 + i * 3},
         {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 17 + i * 3}                              
    ])

    skip_36 = x

    # Layer 37 => 40
    x = self._conv_block(x, [
         {"filter": 512, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 37},
         {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 38},
         {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 39}                              
    ])

    # Layer 41 => 61
    x = self._conv_block(x, [
         {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 41 + i * 3},
         {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 42 + i * 3}                              
    ])

    skip_61 = x

    # Layer 62 => 65
    x = self._conv_block(x, [
         {"filter": 1024, "kernel": 3, "stride": 2, "bnorm": True, "leaky": True, "layer_idx": 62},
         {"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 63},
         {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 64}                              
    ])

    # Layer 66 => 74
    for i in range(3):
      x = self._conv_block(x, [
         {"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 66 + i * 3},
         {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 67 + i * 3}                              
      ])

    # Layer 75 => 79
    x = self._conv_block(x, [
         {"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 75},
         {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 76},
         {"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 77},
         {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 78},
         {"filter": 512, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 79}                              
    ], skip=False)

    # Layer 80 => 82
    yolo_82 = self._conv_block(x, [
         {"filter": 1024, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 80},
         {"filter": 255, "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 81}                             
    ], skip=False)

    # Layer 83 => 86
    x = self._conv_block(x, [
         {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 84}                         
    ], skip=False)

    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_61])

    # Layer 87 => 91
    x = self._conv_block(x, [
         {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 87},
         {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 88},
         {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 89},
         {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 90},
         {"filter": 256, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 91}                              
    ], skip=False)

    # Layer 92 => 94
    yolo_94 = self._conv_block(x, [
         {"filter": 512, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 92},
         {"filter": 255, "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 93}                             
    ], skip=False)

    # Layer 95 => 98
    x = self._conv_block(x, [
         {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 96}                             
    ], skip=False)

    x = UpSampling2D(2)(x)
    x = Concatenate()([x, skip_36])

    # Layer 99 => 106
    yolo_106 = self._conv_block(x, [
         {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 99},
         {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 100},
         {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 101},
         {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 102},
         {"filter": 128, "kernel": 1, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 103},
         {"filter": 256, "kernel": 3, "stride": 1, "bnorm": True, "leaky": True, "layer_idx": 104},
         {"filter": 255, "kernel": 1, "stride": 1, "bnorm": False, "leaky": False, "layer_idx": 105}                          
    ], skip=False)

    return Model(inputs=input_image, outputs=[yolo_82, yolo_94, yolo_106])

  def _load_yolo(self):
    """
    This method creates the architecture, loads the weights, and instantiates a trained YOLO model in a format TensorFlow understands.
    """
    model = self._make_yolov3_architecture()
    weight_reader = WeightReader(self.weights_path)
    weight_reader.load_weights(model)
    model.save("model.h5")

    model = load_model("model.h5")

    return model

  @staticmethod
  def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))

  def _decode_net_output(self, network_output, anchors, obj_thresh, network_height, network_width):
      """
      This method decodes the candidate bounding boxes and class predictions produced by YOLO
      """
      grid_height, grid_width = network_output.shape[:2]
      nb_box = 3
      network_output = network_output.reshape((grid_height, grid_width, nb_box, -1))

      boxes = []
      network_output[..., :2] = self._sigmoid(network_output[..., :2])
      network_output[..., 4:] = self._sigmoid(network_output[..., 4:])
      network_output[..., 5:] = (network_output[..., 4] [..., np.newaxis] * network_output[..., 5:)
      network_output[..., 5:] = network_output[..., 5:] > obj_thresh

      
