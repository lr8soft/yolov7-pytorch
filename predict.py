# -----------------------------------------------------------------------#
#   predict.py将单张图片预测、摄像头检测、FPS测试和目录遍历检测等功能
#   整合到了一个py文件中，通过指定mode进行模式的修改。
# -----------------------------------------------------------------------#
import multiprocessing
import time
import cv2
import numpy as np
from PIL import Image

from yolo import YOLO, YOLO_ONNX


def convert_to_xywh(bbox):
    """将bbox(top, left, bottom, right)转换为bbox(xmin, ymin, width, height)"""
    ymin, xmin, ymax, xmax = bbox

    new_xmin = xmin
    new_ymin = ymin
    height = max(0, ymax - ymin)
    width = max(0, xmax - xmin)
    return new_xmin, new_ymin, width, height


class TrackerWrapper:
    OBJECT_INDEX = 0

    def __init__(self):
        self.tracker = None
        self.box = None
        self.index = TrackerWrapper.OBJECT_INDEX
        TrackerWrapper.OBJECT_INDEX += 1

    def init(self, image, bbox):
        self.box = bbox
        self.tracker = cv2.TrackerKCF_create()

        self.tracker.init(image, bbox)

    def update(self, image):
        success, box = self.tracker.update(image)
        if success:
            self.box = box
        return success, box

    def check_is_same(self, bbox, iou_threshold=0.5):
        """检查bbox是否和当前tracker的box一致"""
        if self.box is None:
            return False
        return TrackerWrapper.calc_bbox_iou(self.box, bbox) > iou_threshold

    def get_last_box(self):
        return self.box

    def get_index(self):
        return self.index

    @staticmethod
    def calc_bbox_iou(box_a, box_b):
        """通过bbox(xmin, ymin, width, height)计算iou，
        """
        # 计算相交矩形的左上角和右下角坐标
        x1 = max(box_a[0], box_b[0])
        y1 = max(box_a[1], box_b[1])
        x2 = min(box_a[0] + box_a[2], box_b[0] + box_b[2])
        y2 = min(box_a[1] + box_a[3], box_b[1] + box_b[3])
        # 计算相交矩形的面积
        inter_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        # 计算并集面积
        box_a_area = (box_a[2] + 1) * (box_a[3] + 1)
        box_b_area = (box_b[2] + 1) * (box_b[3] + 1)
        union_area = box_a_area + box_b_area - inter_area

        # 计算iou
        iou = inter_area / union_area

        return iou

    @staticmethod
    def check_bbox_similar(box_a, box_b):
        """通过bbox(xmin, ymin, width, height)计算俩bbox面积比，越大越相似，"""
        box_a_area = (box_a[2] + 1) * (box_a[3] + 1)
        box_b_area = (box_b[2] + 1) * (box_b[3] + 1)

        ratio = box_a_area / box_b_area
        if ratio > 1:
            ratio = 1 / ratio

        return ratio


class TrackerWrapperManager:
    def __init__(self):
        self.trackers = []

    def update_by_frame(self, frame):
        """返回更新后的结果和tracker的更新结果"""
        update_result = []
        del_trackers = []
        # 先更新tracker
        for tracker in self.trackers:
            success, box = tracker.update(frame)
            if success:
                update_result.append(box)
            else:
                del_trackers.append(tracker)

        # 通过check_is_same检查是否有重合的tracker
        for tracker in self.trackers:
            for tracker2 in self.trackers:
                if tracker is tracker2:
                    continue

                if tracker.check_is_same(tracker2.get_last_box()) and tracker2 not in del_trackers:
                    del_trackers.append(tracker2)

        # 删除更新失败的tracker
        for tracker in del_trackers:
            self.trackers.remove(tracker)

        return update_result

    def update_by_detections(self, detected_frame, bbox_data):
        update_result = []
        self.trackers = []
        # 然后根据检测结果添加tracker
        if len(bbox_data) > 0:
            tracker_update_dict = {}
            for detected_info in bbox_data:
                _, _, detected_bbox = detected_info
                # 将bbox转换为x y w h
                detected_bbox = convert_to_xywh(detected_bbox)
                update_result.append(detected_bbox)

                tracker = TrackerWrapper()
                tracker.init(detected_frame, detected_bbox)
                self.trackers.append(tracker)

        return update_result


def read_frame_process(cap_url, g_queue: multiprocessing.Queue):
    capture = cv2.VideoCapture(cap_url)
    capture.set(cv2.CAP_PROP_FPS, 24)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    sleep_interval = 1.0 / 24

    last_sec_time = time.time()
    now_frame = 0
    while True:
        current_time = time.time()
        ref, frame = capture.read()
        if not ref:
            break

        # 还在1s内
        if current_time - last_sec_time < 1:
            now_frame += 1
        else:
            # 重置时间
            last_sec_time = current_time
            now_frame = 0

        while g_queue.qsize() > 24:
            if not g_queue.empty():
                if g_queue.qsize() > 24:
                    try:
                        g_queue.get_nowait()
                    except:
                        continue

        if now_frame < 24:
            put_success = False
            while not put_success:
                try:
                    g_queue.put_nowait(frame)
                    put_success = True
                except:
                    continue

        time.sleep(sleep_interval)


def predict_frame_process(need_predict_queue: multiprocessing.Queue, detection_queue: multiprocessing.Queue):
    model = YOLO(model_path="logs/best_epoch_weights.pth", classes_path="model_data/cls_classes.txt", confidence=0.5)
    tracker_manager = TrackerWrapperManager()
    sleep_time = 1.0 / 24.0
    last_predict_time = time.time()
    while True:

        src_frame = None
        while not need_predict_queue.empty():
            try:
                src_frame = need_predict_queue.get_nowait()
            except:
                continue

            # 转成Image
            frame_pil = Image.fromarray(np.uint8(src_frame))

            current_time = time.time()
            # 记录开始预测前的时间
            frame_predict_start_time = time.time()

            result_from_predict = False
            # 预测时间到
            if current_time - last_predict_time > 1:
                image, detections = model.detect_image(frame_pil, False)
                if detections is None:
                    detections = []
                # 通过预测的数据更新跟踪器的状态
                boxes = tracker_manager.update_by_detections(src_frame, detections)
                last_predict_time = current_time
                result_from_predict = True
            else:
                boxes = tracker_manager.update_by_frame(src_frame)

            # 记录这一帧预测的时间
            frame_predict_use_time = time.time() - frame_predict_start_time
            # 计算理论上一秒钟可以预测的帧数
            expect_frame_per_sec = 1.0 / frame_predict_use_time if frame_predict_use_time > 0 else 24
            detection_queue.put((int(expect_frame_per_sec), src_frame, boxes, result_from_predict))



if __name__ == "__main__":
    # ----------------------------------------------------------------------------------------------------------#
    #   mode用于指定测试的模式：
    #   'predict'           表示单张图片预测，如果想对预测过程进行修改，如保存图片，截取对象等，可以先看下方详细的注释
    #   'video'             表示视频检测，可调用摄像头或者视频进行检测，详情查看下方注释。
    #   'fps'               表示测试fps，使用的图片是img里面的street.jpg，详情查看下方注释。
    #   'dir_predict'       表示遍历文件夹进行检测并保存。默认遍历img文件夹，保存img_out文件夹，详情查看下方注释。
    #   'heatmap'           表示进行预测结果的热力图可视化，详情查看下方注释。
    #   'export_onnx'       表示将模型导出为onnx，需要pytorch1.7.1以上。
    #   'predict_onnx'      表示利用导出的onnx模型进行预测，相关参数的修改在yolo.py_423行左右处的YOLO_ONNX
    # ----------------------------------------------------------------------------------------------------------#
    mode = "video"
    # -------------------------------------------------------------------------#
    #   crop                指定了是否在单张图片预测后对目标进行截取
    #   count               指定了是否进行目标的计数
    #   crop、count仅在mode='predict'时有效
    # -------------------------------------------------------------------------#
    crop = False
    count = False
    # ----------------------------------------------------------------------------------------------------------#
    #   video_path          用于指定视频的路径，当video_path=0时表示检测摄像头
    #                       想要检测视频，则设置如video_path = "xxx.mp4"即可，代表读取出根目录下的xxx.mp4文件。
    #   video_save_path     表示视频保存的路径，当video_save_path=""时表示不保存
    #                       想要保存视频，则设置如video_save_path = "yyy.mp4"即可，代表保存为根目录下的yyy.mp4文件。
    #   video_fps           用于保存的视频的fps
    #
    #   video_path、video_save_path和video_fps仅在mode='video'时有效
    #   保存视频时需要ctrl+c退出或者运行到最后一帧才会完成完整的保存步骤。
    # ----------------------------------------------------------------------------------------------------------#
    video_path = "road_test.mp4"
    video_save_path = ""
    video_fps = 25.0
    # ----------------------------------------------------------------------------------------------------------#
    #   test_interval       用于指定测量fps的时候，图片检测的次数。理论上test_interval越大，fps越准确。
    #   fps_image_path      用于指定测试的fps图片
    #
    #   test_interval和fps_image_path仅在mode='fps'有效
    # ----------------------------------------------------------------------------------------------------------#
    test_interval = 100
    fps_image_path = "img/street.jpg"
    # -------------------------------------------------------------------------#
    #   dir_origin_path     指定了用于检测的图片的文件夹路径
    #   dir_save_path       指定了检测完图片的保存路径
    #
    #   dir_origin_path和dir_save_path仅在mode='dir_predict'时有效
    # -------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path = "img_out/"
    # -------------------------------------------------------------------------#
    #   heatmap_save_path   热力图的保存路径，默认保存在model_data下
    #
    #   heatmap_save_path仅在mode='heatmap'有效
    # -------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    # -------------------------------------------------------------------------#
    #   simplify            使用Simplify onnx
    #   onnx_save_path      指定了onnx的保存路径
    # -------------------------------------------------------------------------#
    simplify = True
    onnx_save_path = "model_data/models.onnx"

    if mode != "predict_onnx":
        yolo = None
    else:
        yolo = YOLO_ONNX()

    # 只剩video模式了
    if mode == "video":
        frame_queue = multiprocessing.Queue()

        rtp_process = multiprocessing.Process(target=read_frame_process, args=(video_path, frame_queue,))
        rtp_process.start()

        predict_frame_queue = multiprocessing.Queue()
        predict_detection_queue = multiprocessing.Queue()
        predict_process = multiprocessing.Process(target=predict_frame_process,
                                                  args=(predict_frame_queue, predict_detection_queue,))
        predict_process.start()

        tracker_manager = TrackerWrapperManager()

        fps = 0.0
        last_predict_time = time.time()
        sleep_time = 1.0 / 20.0
        # 是否预测过
        has_predict = False
        # 预测的最大帧数
        predict_max_frame = 24
        while True:
            t1 = time.time()
            # 读取某一帧
            frame = frame_queue.get()

            if frame is None:
                continue

            # 转格式，BGRtoRGB
            source_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # 送入预测队列
            predict_frame_queue.put(source_frame)

            # 这里要根据预测进程，检测预测线程最多能预测多少帧，如果超过这个帧数，就丢弃掉早期的帧
            while predict_frame_queue.qsize() > predict_max_frame:
                if not predict_frame_queue.empty():
                    try:
                        predict_frame_queue.get_nowait()
                    except:
                        pass

            # 有检测结果了
            if not predict_detection_queue.empty():
                max_frame, source_frame, boxes, result_from_predict = predict_detection_queue.get()
                # 限制最大帧数
                predict_max_frame = min(max_frame, 24)
                # 根据检测到的结果，开画！
                for box in boxes:
                    (x, y, w, h) = [int(v) for v in box]
                    cv2.rectangle(source_frame, (x, y), (x + w, y + h), (255, 255, 0), 2)

                    if result_from_predict:
                        source_frame = cv2.putText(source_frame, "Predict result",
                                                   (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 2)
                    else:
                        source_frame = cv2.putText(source_frame, "Tracking result",
                                                   (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            else:
                continue

            frame = source_frame
            # RGBtoBGR满足opencv显示格式
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            frame = cv2.putText(frame, "expect_frame= %.2f" % predict_max_frame,
                                (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # resize到640x480
            frame = cv2.resize(frame, (640, 480))

            cv2.imshow("video", frame)
            c = cv2.waitKey(1) & 0xff
            time.sleep(sleep_time)

        print("Video Detection Done!")
        cv2.destroyAllWindows()
