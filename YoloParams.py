class YoloParams:
    # Extracting layer parameters
    # Magic numbers are copied from YOLOv3 samples
    def __init__(self, param, side):
        self.num = 3 if 'num' not in param else int(param['num'])
        self.coords = 4 if 'coords' not in param else int(param['coords'])
        self.classes = 80 if 'classes' not in param else int(param['classes'])
        #self.anchors = [10.0, 13.0, 16.0, 30.0, 33.0, 23.0, 30.0, 61.0, 62.0, 45.0, 59.0, 119.0, 116.0, 90.0, 156.0,
        #            198.0, 373.0, 326.0] if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
        self.anchors = [10.0,14.0,  23.0,27.0,  37.0,58.0,  81.0,82.0,  135.0,169.0,  344.0,319.0] \
            if 'anchors' not in param else [float(a) for a in param['anchors'].split(',')]
        self.mask = [3,4,5]

        if 'mask' in param:
            mask = [int(idx) for idx in param['mask'].split(',')]
            self.num = len(mask)

            maskedAnchors = []
            for idx in mask:
                maskedAnchors += [self.anchors[idx * 2], self.anchors[idx * 2 + 1]]
            self.anchors = maskedAnchors

        self.side = side
        self.isYoloV3 = 'mask' in param  # Weak way to determine but the only one.
