import argparse
from mvavatar import MVAvatar
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
import cv2
import numpy as np
from PIL import Image
import os

def human_matting_fn(img_path_list):
    # segment image
    algo_matting = pipeline(Tasks.portrait_matting, model='damo/cv_unet_image-matting')
    for img_path in img_path_list:
        result = algo_matting(img_path)
        rgba = result[OutputKeys.OUTPUT_IMG]
        mask = rgba[:, :, 3]
        color = rgba[:, :, :3]
        alpha = mask / 255
        # bk = np.ones_like(color) * 255
        bk = np.zeros_like(color) * 255
        color = color * alpha[:, :, np.newaxis] + bk * (1 - alpha[:, :, np.newaxis])
        color = color.astype(np.uint8)
        cv2.imwrite(img_path, color)



def split_image(img, n_row=1, n_col=7):
    # PIL to numpy
    img = np.array(img)
    # split image into n_row * n_col
    h, w, c = img.shape
    assert h % n_row == 0
    assert w % n_col == 0
    h_per_row = h // n_row
    w_per_col = w // n_col
    imgs = []
    for i in range(n_row):
        for j in range(n_col):
            # numpy to PIL
            imgs.append(Image.fromarray(img[i * h_per_row:(i + 1) * h_per_row, j * w_per_col:(j + 1) * w_per_col, :]))
    return imgs

def format_image(path, n_col=2):
    # n_col = 2
    w_crop = 100
    h_crop = 20
    print('format image %s' % path)
    img = cv2.imread(path)[..., ::-1]
    h, w, c = img.shape
    w = int(w / n_col)
    images = split_image(img, n_row=1, n_col=n_col)
    scale = h / 480
    image_list = []
    idx = 0
    outpath_list = []
    for img in images:
        img_full = img.crop((-w_crop * scale, -h_crop * scale, w + w_crop * scale, h + h_crop * scale))
        img_full = img_full.resize((1024, 1024))
        img_np = np.array(img_full)[:, :, ::-1]
        image_list.append(img_np)
        if idx == 0:
            outpath = path
        elif idx == 1:
            outpath = path[:-4] + '_back.png'
        elif idx == 2:
            outpath = path[:-4] + '_side.png'
        else:
            idx += 1
            continue
        # resize image
        img_np = cv2.resize(img_np, (512, 512), interpolation=cv2.INTER_AREA)
        cv2.imwrite(outpath, img_np)
        outpath_list.append(outpath)
        idx += 1

    # # concat image
    # img_np = np.concatenate(image_list, axis=1)
    # cv2.imwrite(path, img_np)
    return outpath_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', default='', type=str)
    parser.add_argument('--outpath', default='', type=str)
    parser.add_argument('--n_view', default=2, type=int)

    args = parser.parse_args()
    outpath = args.outpath
    text = args.text
    n_view = args.n_view

    if not text.endswith('resolution'):
        text = text+ ' white background, simple background, best quality, high resolution'

    # model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), 'models')
    model_dir = '/data/qingyao/neuralRendering/mycode/pretrainedModel/3dhuman_gen/En3D/lib/synthesis_2d/models'
    model = MVAvatar.from_pretrained(model_dir, device='cuda', n_view=n_view)

    height = int(480 * 1.8)
    std_width = int(2240 * 1.8 / 7)
    width = std_width * n_view

    image = model.inference(
        text, width=width, height=height, seed=-1,
    )

    if not os.path.exists(os.path.dirname(outpath)):
        os.makedirs(os.path.dirname(outpath))

    image.save(outpath)
    print('generate mvavatr image finished!')

    outpath_list = format_image(outpath, n_col=n_view)
    print('format image finished!')

    human_matting_fn(outpath_list)
    print('segment image finished!')
    print('save image to {}'.format(outpath))


