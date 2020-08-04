import argparse
import cv2
from math import exp as exp
from time import time
from imutils.video import FPS
from inference import Network
from YoloParams import YoloParams


# Build the argument parser and parse the arguments
def get_args():
    """
    Obtains the application arguments from the command line.
    """
    ap = argparse.ArgumentParser("Run tiny YOLOv3 inference on an input camera stream or input video")

    # Adds required and optional argument groups
    ap._action_groups.pop()
    required = ap.add_argument_group('required arguments')
    optional = ap.add_argument_group('optional arguments')

    # Sets the individual arguments with respect to the relevant argument group
    required.add_argument("--m", "--model", help="Required: Path to .xml file with trained YOLOv3 model",
                          required=True, type=str)
    optional.add_argument("--i", "--input_file", help="Optional: Path to a video file. If not provided, \
                          camera input is used", default='CAM', type=str)
    optional.add_argument("--l", "--label_file", help="Optional: Path to a file of object class labels", type=str)
    optional.add_argument("--d", "--device_name", help="Optional: Name of the device to do inference on. CPU (default),
                          GPU, FPGA, HDDL or MYRIAD", default='CPU', type=str)
    optional.add_argument("--t", "--threshold", help="Optional: Confidence threshold for filtering of object \
                          detections", default=0.5, type=float)
    optional.add_argument("--o", "--overlap", help="Optional: Maximum permissible bounding box overlap", default=0.3,
                          type=float)
    args = ap.parse_args()

    return args


def entry_index(side, coord, classes, location, entry):
    side_power_2 = side ** 2
    n = location // side_power_2
    loc = location % side_power_2

    return int(side_power_2 * (n * (coord + classes + 1) + entry) + loc)


def scale_bbox(x, y, h, w, class_id, confidence, h_scale, w_scale):
    """
    Takes (x, y) to be the centre point of the bounding box.
    Scales the box according to the width and height scaling parameters.
    """
    xmin = int((x - w / 2) * w_scale)
    ymin = int((y - h / 2) * h_scale)
    xmax = int(xmin + w * w_scale)
    ymax = int(ymin + h * h_scale)

    return dict(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, class_id=class_id, confidence=confidence)


def intersection_over_union(box_1, box_2):
    """
    Determines extent of aerial overlap between bounding boxes 1 and 2
    """
    width_of_overlap_area = min(box_1['xmax'], box_2['xmax']) - max(box_1['xmin'], box_2['xmin'])
    height_of_overlap_area = min(box_1['ymax'], box_2['ymax']) - max(box_1['ymin'], box_2['ymin'])
    if width_of_overlap_area < 0 or height_of_overlap_area < 0:
        area_of_overlap = 0
    else:
        area_of_overlap = width_of_overlap_area * height_of_overlap_area

    box_1_area = (box_1['ymax'] - box_1['ymin']) * (box_1['xmax'] - box_1['xmin'])
    box_2_area = (box_2['ymax'] - box_2['ymin']) * (box_2['xmax'] - box_2['xmin'])
    area_of_union = box_1_area + box_2_area - area_of_overlap

    if area_of_union == 0:
        return 0
    return area_of_overlap / area_of_union


def parse_yolo_region(blob, resized_image_shape, original_im_shape, params, threshold):
    """
    Takes the YOLO output blob, parses validated regions and
    adds scaled bounding boxes to the list of detected objects
    """

    # Validating output parameters
    _, _, out_blob_h, out_blob_w = blob.shape
    assert out_blob_w == out_blob_h, "Invalid size of output blob. It should be in NCHW layout and height should " \
                                     "be equal to width. Current height = {}, current width = {}" \
                                     "".format(out_blob_h, out_blob_w)

    # Extracting layer parameters
    orig_im_h, orig_im_w = original_im_shape
    resized_image_h, resized_image_w = resized_image_shape
    objects = list()
    predictions = blob.flatten()
    side_square = params.side * params.side

    #  Parsing YOLO region output
    for i in range(side_square):
        row = i // params.side
        col = i % params.side
        for n in range(params.num):
            obj_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, params.coords)
            scale = predictions[obj_index]
            if scale < threshold:
                continue
            box_index = entry_index(params.side, params.coords, params.classes, n * side_square + i, 0)
            # Network produces location predictions in absolute coordinates of feature maps.
            # Scale it to relative coordinates.
            x = (col + predictions[box_index + 0 * side_square]) / params.side
            y = (row + predictions[box_index + 1 * side_square]) / params.side
            # Value for exp is very big number in some cases, so check for overflow
            try:
                w_exp = exp(predictions[box_index + 2 * side_square])
                h_exp = exp(predictions[box_index + 3 * side_square])
            except OverflowError:
                continue
            # Depending on topology, frame size is to be normalised by feature maps (<YOLOv3) or input shape (=YOLOv3)
            w = w_exp * params.anchors[2 * n] / (resized_image_w if params.isYoloV3 else params.side)
            h = h_exp * params.anchors[2 * n + 1] / (resized_image_h if params.isYoloV3 else params.side)
            for j in range(params.classes):
                class_index = entry_index(params.side, params.coords, params.classes, n * side_square + i,
                                          params.coords + 1 + j)
                confidence = scale * predictions[class_index]
                if confidence < threshold:
                    continue
                objects.append(scale_bbox(x=x, y=y, h=h, w=w, class_id=j, confidence=confidence,
                                          h_scale=orig_im_h, w_scale=orig_im_w))

    return objects


def infer_video(args):
    """
    Take input video or camera stream and perform asynchronous object detection inference
    using tiny YOLOv3 on a frame-by-frame basis
    """
    # Initialize the OpenVINO Inference Engine
    net = Network()

    # Read object class labels (here: coco.names)
    if args.l:
        with open(args.l, 'r') as f:
            labels_map = [x.strip() for x in f]
    else:
        labels_map = None

    # Load the Deep Learning Computer Vision model into the Inference Engine
    net.load_model(args.m, args.d)
    n, c, h, w = net.get_input_shape()

    # Window for stream and video inference result display
    window_name = "Processing video input"
    cv2.namedWindow(window_name)

    # Set up argument for local camera frame capture, if applicable
    if args.i == 'CAM':
        args.i = 0

    # Set up OpenCV video capturing object for local camera or video file processing
    capture = cv2.VideoCapture(args.i)
    capture.open(args.i)

    # Set up OpenCV video writer object for output video generation using XVID codec
    file_in = "output.avi"
    codec = cv2.VideoWriter_fourcc("M", "P", "4", "2") # Raspbian setting after FFMPEG 1.4 installation
    frame_rate = 30
    width = int(capture.get(3))
    height = int(capture.get(4))
    resolution = (width, height)
    file_out = cv2.VideoWriter(file_in, codec, frame_rate, resolution)

    # Process input frames until end of video or process is exited by escape keystroke
    fps = FPS().start()
    while capture.isOpened():
        flag, frame = capture.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        # Pre-process input frame as required by Deep Learning model
        # 1. Resize to shape expected by Deep Learning model
        # 2. Transpose data layout from HWC (height, width, channel) to CHW
        # 3. Reshape the frame to add a "batch" of 1 at the start
        pre_frame = cv2.resize(frame, (w, h))
        pre_frame = pre_frame.transpose((2, 0, 1))
        pre_frame = pre_frame.reshape((n, c, h, w))

        # Start inference on the pre-processed frame and compute inference duration
        start_time = time()
        net.async_inference(pre_frame)
        detection_time = time() - start_time

        # Obtain the inference result
        objects = list()
        if net.wait() == 0:
            output = net.extract_output()

            for layer_name, out_blob in output.items():
                out_blob = out_blob.reshape(net.network.layers[net.network.layers[layer_name].parents[0]].shape)
                layer_params = YoloParams(net.network.layers[layer_name].params, out_blob.shape[2])
                objects += parse_yolo_region(out_blob, pre_frame.shape[2:], frame.shape[:-1], layer_params, args.t)

        # Filter out overlapping bounding boxes with respect to the IoU parameter
        objects = sorted(objects, key=lambda obj : obj['confidence'], reverse=True)
        for i in range(len(objects)):
            if objects[i]['confidence'] == 0:
                continue
            for j in range(i + 1, len(objects)):
                if intersection_over_union(objects[i], objects[j]) > args.o:
                    objects[j]['confidence'] = 0

        # Draw objects with respect to probability threshold parameter
        objects = [obj for obj in objects if obj['confidence'] >= args.t]
        origin_im_size = frame.shape[:-1]
        for obj in objects:
            if obj['xmax'] > origin_im_size[1] or obj['ymax'] > origin_im_size[0] or obj['xmin'] < 0 or obj['ymin'] < 0:
                continue
            colour = (int(min(obj['class_id'] * 12.5, 255)), min(obj['class_id'] * 7, 255), min(obj['class_id'] * 5, 255))
            det_label = labels_map[obj['class_id']] if labels_map and len(labels_map) >= obj['class_id'] else \
                str(obj['class_id'])
            cv2.rectangle(frame, (obj['xmin'], obj['ymin']), (obj['xmax'], obj['ymax']), colour, 2)
            cv2.putText(frame, det_label + ' ' + str(round(obj['confidence'] * 100, 1)) + ' %',
                        (obj['xmin'], obj['ymin'] - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, colour, 1)
            cv2.putText(frame, "Inference time: {:.3f} ms".format(detection_time * 1e3), (15, 15),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)

        file_out.write(frame)
        cv2.imshow(window_name, frame)
        if key_pressed == 27:
            break

        # Update frames-per-second information
        fps.update()

    fps.stop()
    print("Elapsed time: {:.2f}".format(fps.elapsed()))
    print("Approximate FPS: {:.2f}".format(fps.fps()))

    file_out.release()
    capture.release()
    cv2.destroyAllWindows()


def main():
    args = get_args()
    infer_video(args)


if __name__ == "__main__":
    main()
