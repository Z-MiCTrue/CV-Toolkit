import os
import cv2


def read_file_num(path):
    files = os.listdir(path)  # 读入文件夹
    num = len(files)
    return num


def Videos_to_Pictures(videos_path, saveDir='pictures/'):
    if not os.path.exists(saveDir):
        os.makedirs(saveDir)
    vc = cv2.VideoCapture(videos_path)
    picture_name = 0
    if vc.isOpened():
        ret, frame = vc.read()
    else:
        ret = False
        frame = None
    while ret:
        cv2.imwrite(saveDir + str(picture_name) + '.jpg', frame)
        picture_name += 1
        ret, frame = vc.read()
    frame_count = vc.get(5)
    print('{}: {}'.format('The video frame rate is', frame_count))
    print('{}: {}{}'.format('All is divided in to', picture_name, 'frames'))
    print('Write over!')
    vc.release()


def Pictures_to_Videos(picture_path, fps):
    size = cv2.imread(picture_path + '0.jpg').shape[:2]
    size = size[::-1]
    video = cv2.VideoWriter("output.avi", cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), fps, size)  # MPEG-4 编码
    for i in range(read_file_num(picture_path)):      
        image_path = picture_path + str(i) + '.jpg'
        img = cv2.imread(image_path)
        video.write(img)
    print('Write over!')
    video.release()


if __name__ == '__main__':
    # Videos_to_Pictures('1.mp4')
    Pictures_to_Videos('pictures/', 10)
