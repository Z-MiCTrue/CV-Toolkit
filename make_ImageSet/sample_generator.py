from xml.etree.ElementTree import Element, SubElement, ElementTree
from copy import deepcopy
import xml.dom.minidom

import numpy as np
import cv2
import os

from utils_image import enhance, general_transform


def make_file_list(folder_name: str):
    file_list = []
    for root, dirs, files in os.walk(folder_name):  # 工作目录, 子目录, 文件
        for file_name in files:
            file_list.append(os.path.join(folder_name, file_name))
    return file_list


# sort data by a
def zip_sort(a, b, c, big2small):
    abc_zipped = zip(a, b, c)
    sorted_abc_zipped = sorted(abc_zipped, key=(lambda x: x[0]), reverse=big2small)  # False: 升序
    a, b, c = [x for x in zip(*sorted_abc_zipped)]
    return a, b, c


# Appropriate distribution of validation and training sets
def category_assign(cache_data: list, class_distribution: np.ndarray, label_list: list):
    # 打印各类别数
    print(f'class distribution{label_list}: {np.sum(class_distribution, axis=0)}')
    # 计算目标验证图片个数以及每类验证集个数
    val_num_img = np.maximum(round(len(cache_data) / 10), 1)  # 最低为1
    val_num_class = np.maximum(round(len(cache_data) / 10 / len(label_list)), 1)  # 最低为1
    class_distribution = np.minimum(class_distribution, val_num_class)  # 抑制单一类别过多
    # 按贡献函数依次选入验证集
    cache_data_sorted = []
    val_count = np.zeros(len(label_list), dtype=int)
    for i in range(val_num_img):
        sort_key = np.sum(np.abs(np.minimum(class_distribution + val_count, val_num_class) -
                                 np.minimum(val_count, val_num_class)), axis=1)
        sort_key, class_distribution, cache_data = zip_sort(sort_key, class_distribution, cache_data,
                                                            big2small=True)
        sort_key, class_distribution, cache_data = np.array(sort_key), np.array(class_distribution), list(cache_data)
        cache_data_sorted.insert(-1, cache_data[0])
        val_count += np.array(class_distribution[0])
        class_distribution, cache_data = class_distribution[1:], cache_data[1:]
    cache_data_sorted = cache_data + cache_data_sorted
    return cache_data_sorted


# Load label into memory
# format: [[img_name, [[label, [x_1, y_1, x_2, y_2]]], [height, width, channel]]]
def load_label(file_name_list: list, label_list: list, img_format: str, special_size: tuple = None):
    if os.path.exists('./cache_data.txt'):
        with open('cache_data.txt', 'r', encoding='utf-8') as cache_data_txt:
            cache_data = cache_data_txt.read()
        cache_data = eval(cache_data)
        cache_data_changed = False
    else:
        cache_data = []
        class_distribution = np.zeros((len(file_name_list), len(label_list)), dtype=int)
        # transform vox xml label
        for j, xml_file in enumerate(file_name_list):
            img_name = xml_file.split('/')[-1].split('.')[0] + img_format
            cache_data.append([img_name, [], None])
            # 打开xml文档
            xml_dom = xml.dom.minidom.parse(xml_file)
            # 读取锚框
            for aim_object in xml_dom.getElementsByTagName('object'):
                label = aim_object.getElementsByTagName('name')[0].firstChild.data
                x_1 = float(aim_object.getElementsByTagName('xmin')[0].firstChild.data)
                y_1 = float(aim_object.getElementsByTagName('ymin')[0].firstChild.data)
                x_2 = float(aim_object.getElementsByTagName('xmax')[0].firstChild.data)
                y_2 = float(aim_object.getElementsByTagName('ymax')[0].firstChild.data)
                cache_data[-1][1].append([label, [x_1, y_1, x_2, y_2]])
                class_distribution[j][label_list.index(label)] += 1
            # 读取尺寸
            assert os.path.exists(f'./images/{img_name}') is True
            img = cv2.imread(f'./images/{img_name}', 1)
            cache_data[-1][2] = list(img.shape[:3])
        # 包含类别数排序
        cache_data = category_assign(cache_data, class_distribution, label_list)
        cache_data_changed = True
    # 放缩至目标尺寸
    if special_size is not None:
        for j, meta in enumerate(cache_data):
            img_name = meta[0]
            assert os.path.exists(f'./images/{img_name}') is True
            img = cv2.imread(f'./images/{img_name}', 1)
            points_all = []
            for label_bbox in meta[1]:
                points_all.append(label_bbox[1])
            img, points_res = general_transform.img_resize(img, special_size, keep_ratio=True, points=points_all)
            for i, point_res in enumerate(points_res):
                cache_data[j][1][i][1] = point_res
            cv2.imwrite(f'./images/{img_name}', img)
            # 重写尺寸
            cache_data[j][2] = list(img.shape[:3])
        cache_data_changed = True
    # 记录数据
    if cache_data_changed:
        with open('cache_data.txt', 'w', encoding='utf-8') as cache_data_txt:
            cache_data_str = str(cache_data)
            cache_data_txt.write(cache_data_str)
    return cache_data


def img_enhance(cache_data: list, config):
    print("enhance images, please wait...")
    new_cache_data = deepcopy(cache_data)
    for meta in cache_data:
        # 读图
        assert os.path.exists(f'./images/{meta[0]}') is True
        img = cv2.imread(f'./images/{meta[0]}', 1)
        height, width, channel = meta[2]
        # ---旋转与模糊--- #
        # 编码
        labels = []
        points = []
        for point in meta[1]:
            labels.append(point[0])
            labels.append(point[0])
            points.append(point[1][:2])
            points.append(point[1][2:])
        # 随机旋转 +- degree
        angle = np.random.choice((1, -1)) * config.rotate_degree
        angle = np.maximum(angle, -90)  # 最小-90
        angle = np.minimum(angle, 90)  # 最大90
        img_E1, points = enhance.img_rotate(img, width=width, height=height, points=points, angle=angle)
        # 解码
        label_bbox = []
        for i in range(0, len(points), 2):
            x_1, y_1 = points[i]
            x_2, y_2 = points[i + 1]
            if config.rotate_keep_all:
                bigger_w = (x_2 - x_1) * np.cos(np.radians(angle)) + (y_2 - y_1) * np.sin(np.radians(np.abs(angle)))
                bigger_h = (y_2 - y_1) * np.cos(np.radians(angle)) + (x_2 - x_1) * np.sin(np.radians(np.abs(angle)))
                centre = [(x_1 + x_2) / 2, (y_1 + y_2) / 2]
                x_1, x_2 = centre[0] - bigger_w / 2, centre[0] + bigger_w / 2
                y_1, y_2 = centre[1] - bigger_h / 2, centre[1] + bigger_h / 2
            label_bbox.append([labels[i], [x_1, y_1, x_2, y_2]])
        img_name = meta[0].split('.')[0] + '_E1.jpg'
        # 模糊
        img_E1 = cv2.GaussianBlur(img_E1, ksize=config.kernel_size, sigmaX=0, sigmaY=0)
        # 添加至数据缓存库
        new_cache_data.append([img_name, label_bbox, meta[2]])
        cv2.imwrite(f'./images/{img_name}', img_E1)
        # ---添加噪声与随机明暗增强--- #
        degree = np.random.choice((1, -1)) * config.fill_light
        img_E2 = enhance.img_expose(img, degree=degree)
        img_E2 = enhance.img_addGauss(img_E2, width=width, height=height, channel=channel, scale=config.noise_scale)
        img_name = meta[0].split('.')[0] + '_E2.jpg'
        # 添加至数据缓存库
        new_cache_data.append([img_name, meta[1], meta[2]])
        cv2.imwrite(f'./images/{img_name}', img_E2)
    return new_cache_data


def bbox_preview(img: np.ndarray, bbox_info: list):
    cv2.rectangle(img, (int(bbox_info[1][0]), int(bbox_info[1][1])),
                  (int(bbox_info[1][2]), int(bbox_info[1][3])), (255, 0, 0), 2)
    cv2.imshow('example', img)
    cv2.waitKey(50)


def make_cascade_classifier(cache_data: list, label_list: list):
    print('make cascade classifier dataset')
    if not os.path.exists('./cascade_classifier_dataset/'):
        os.makedirs('./cascade_classifier_dataset/')
    label_all = np.array([[]] * len(label_list)).tolist()  # 统计面积
    for meta in cache_data:
        label_group = np.array([[]] * len(label_list)).tolist()  # 异化[]地址
        for bbox_info in meta[1]:
            if not os.path.exists(f'./cascade_classifier_dataset/pos_{bbox_info[0]}/'):
                os.makedirs(f'./cascade_classifier_dataset/pos_{bbox_info[0]}/')
            if not os.path.exists(f'./cascade_classifier_dataset/neg_{bbox_info[0]}/'):
                os.makedirs(f'./cascade_classifier_dataset/neg_{bbox_info[0]}/')
            label_group[label_list.index(bbox_info[0])].append(bbox_info[1])
        # neg 制作
        for i, bbox_labeled in enumerate(label_group):
            assert os.path.exists(f'./images/{meta[0]}') is True
            img_neg = cv2.imread(f'./images/{meta[0]}', 1)
            for bbox in bbox_labeled:
                img_neg[round(bbox[1]): round(bbox[3]), round(bbox[0]): round(bbox[2])] = 0
            cv2.imwrite(f'./cascade_classifier_dataset/neg_{label_list[i]}/{meta[0]}', img_neg)
            # 纳入面积统计
            if len(bbox_labeled) > 0:
                label_all[i] += bbox_labeled
    area_resize = np.ones(len(label_list), dtype=np.float32)
    for i, bbox_group in enumerate(label_all):
        bbox_group = np.array(bbox_group)
        area_resize[i] = np.power(np.mean((bbox_group[:, 3] - bbox_group[:, 1]) *
                                          (bbox_group[:, 2] - bbox_group[:, 0])), 0.5)
    # pos 制作
    for meta in cache_data:
        for i, bbox_info in enumerate(meta[1]):
            img_pos = cv2.imread(f'./images/{meta[0]}', 1)
            img_pos = img_pos[round(bbox_info[1][1]): round(bbox_info[1][3]),
                              round(bbox_info[1][0]): round(bbox_info[1][2])]
            label_id = label_list.index(bbox_info[0])
            img_pos = general_transform.img_resize(img_pos,
                                                   (round(area_resize[label_id]), round(area_resize[label_id])),
                                                   keep_ratio=True)
            cv2.imwrite(f'./cascade_classifier_dataset/pos_{bbox_info[0]}/{i}_{meta[0]}', img_pos)


def make_coco(cache_data: list, label_list: list, train_data=True, img_id_start=0, bbox_id_start=0):
    print('make coco dataset')
    if not os.path.exists('./coco_dataset/'):
        os.makedirs('./coco_dataset/')
    # 生成预览框
    cv2.namedWindow('example', cv2.WINDOW_AUTOSIZE)
    # ---写入json--- #
    categories = []
    for category_id, category in enumerate(label_list):
        categories.append({'supercategory': category, 'id': category_id, 'name': category})
    images = []
    annotations = []
    img_id = None
    for img_id_relative, img_info in enumerate(cache_data):
        # 实际image_id
        img_id = img_id_start + img_id_relative
        images.append({'file_name': img_info[0], 'height': img_info[2][0], 'width': img_info[2][1], 'id': img_id})
        # 读图
        assert os.path.exists(f'./images/{img_info[0]}') is True
        img = cv2.imread(f'./images/{img_info[0]}', 1)
        bbox_id = None
        for bbox_id_relative, bbox_info in enumerate(img_info[1]):
            # 实际bbox_id
            bbox_id = bbox_id_start + bbox_id_relative
            annotations.append({'segmentation': [[]],  # 用于语义分割, 目标检测不关注
                                'area': (bbox_info[1][2] - bbox_info[1][0]) * (bbox_info[1][3] - bbox_info[1][1]),
                                'iscrowd': 0,
                                'image_id': img_id,
                                # bbox: [x, y, w, h]
                                'bbox': [bbox_info[1][0], bbox_info[1][1],
                                         bbox_info[1][2] - bbox_info[1][0], bbox_info[1][3] - bbox_info[1][1]],
                                'category_id': label_list.index(bbox_info[0]),
                                'id': bbox_id})
            # 预览图
            bbox_preview(img, bbox_info)
        if bbox_id is not None:
            bbox_id_start = bbox_id + 1
    if img_id is not None:
        img_id_start = img_id + 1
    # 结束预览
    cv2.destroyAllWindows()
    str_write = str({"images": images, "annotations": annotations, "categories": categories}).replace("'", '"')
    if train_data:
        with open('./coco_dataset/train_coco.json', 'w') as coco_out:
            coco_out.write(str_write)
    else:
        with open('./coco_dataset/val_coco.json', 'w') as coco_out:
            coco_out.write(str_write)
    # 返回下一次的起始id
    return img_id_start, bbox_id_start


def make_voc(cache_data: list):
    print('make voc dataset')
    if not os.path.exists('./voc_xml'):
        os.makedirs('./voc_xml')
    # 生成预览框
    cv2.namedWindow('example', cv2.WINDOW_AUTOSIZE)
    for meta in cache_data:
        # ---写入xml--- #
        # root
        root_node = Element('annotation')
        # folder
        folder_node = SubElement(root_node, 'folder')
        folder_node.text = 'images'
        # filename
        filename_node = SubElement(root_node, 'filename')
        filename_node.text = meta[0]
        # source
        source_node = SubElement(root_node, 'source')
        database_node = SubElement(source_node, 'database')
        database_node.text = 'Mitre'
        # size
        size_node = SubElement(root_node, 'size')
        width_node = SubElement(size_node, 'width')
        width_node.text = str(meta[2][1])
        height_node = SubElement(size_node, 'height')
        height_node.text = str(meta[2][0])
        depth_node = SubElement(size_node, 'depth')
        depth_node.text = str(meta[2][2])
        # segmented
        segmented = SubElement(root_node, 'segmented')
        segmented.text = str(0)
        # 读图
        assert os.path.exists('./images/' + meta[0]) is True
        img = cv2.imread('images/' + meta[0], 1)
        for bbox_info in meta[1]:
            # object
            object_node = SubElement(root_node, 'object')
            name_node = SubElement(object_node, 'name')
            name_node.text = bbox_info[0]
            pose_node = SubElement(object_node, 'pose')
            pose_node.text = 'Unspecified'
            truncated_node = SubElement(object_node, 'truncated')
            truncated_node.text = str(0)
            difficult_node = SubElement(object_node, 'difficult')
            difficult_node.text = str(0)
            bndbox_node = SubElement(object_node, 'bndbox')
            xmin_node = SubElement(bndbox_node, 'xmin')
            xmin_node.text = str(int(bbox_info[1][0]))
            ymin_node = SubElement(bndbox_node, 'ymin')
            ymin_node.text = str(int(bbox_info[1][1]))
            xmax_node = SubElement(bndbox_node, 'xmax')
            xmax_node.text = str(int(bbox_info[1][2]))
            ymax_node = SubElement(bndbox_node, 'ymax')
            ymax_node.text = str(int(bbox_info[1][3]))
            # 预览图
            bbox_preview(img, bbox_info)
        # make
        result_tree = ElementTree(root_node)
        # write out xml data
        result_tree.write(f'./voc_xml/{meta[0][:-4]}.xml', encoding='utf-8')
    # 结束预览
    cv2.destroyAllWindows()


def main(args):
    # 获取标签名
    file_name_list = make_file_list(args.folder_name)
    # 加载 & 分配数据集
    dataset_cache = load_label(file_name_list, args.label_list, args.img_format, special_size=args.img_size)
    if args.cascade_classifier:
        if args.image_enhance:
            dataset_cache = img_enhance(dataset_cache, args)
        make_cascade_classifier(dataset_cache, args.label_list)
    if args.coco:
        # 训练集 (取数据集的90%作为验证集)
        cache_data_train = dataset_cache[: round(len(file_name_list) * args.training_ratio)]
        if args.image_enhance:
            cache_data_train = img_enhance(cache_data_train, args)
        img_id_start, bbox_id_start = make_coco(cache_data_train, args.label_list,
                                                train_data=True, img_id_start=0, bbox_id_start=0)
        # 验证集 (取数据集的10%作为验证集)
        cache_data_val = dataset_cache[round(len(file_name_list) * args.training_ratio):]
        if args.image_enhance:
            cache_data_val = img_enhance(cache_data_val, args)
        make_coco(cache_data_val, args.label_list,
                  train_data=False, img_id_start=img_id_start, bbox_id_start=bbox_id_start)
    if args.voc:
        if args.image_enhance:
            dataset_cache = img_enhance(dataset_cache, args)
        make_voc(dataset_cache)
    print('Over!')


class Options:
    def __init__(self):
        # ---base--- #
        # Attention: 新样本需删除缓存
        self.cascade_classifier = True
        self.coco = False
        self.voc = False
        self.label_list = ['H']
        self.img_format = '.jpg'
        self.folder_name = './labels_xml/'
        self.training_ratio = 0.9
        self.image_enhance = True
        self.img_size = None  # w x h e.g.(320, 224)
        # ---others--- #
        # whether to keep the bounding box
        self.rotate_keep_all = True  # careful: |rotate_degree| < 90
        # rotation angle
        self.rotate_degree = 5
        # Gaussian kernel size
        self.kernel_size = (5, 5)
        # noise rate
        self.noise_scale = 5e-1
        # light and dark enhancement
        self.fill_light = 32
        # ---auto--- #
        assert os.path.exists('./images') is True


if __name__ == '__main__':
    use_args = Options()
    main(use_args)
