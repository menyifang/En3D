import os
import imageio
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys
from modelscope.models.cv.face_reconstruction.utils import write_obj
import requests

def list_from_url(url):
    r = requests.get(url)
    list = r.content.decode('utf-8').split('\n')
    list = list[:-1]
    return list

def save_results(result, save_root):
    os.makedirs(save_root, exist_ok=True)

    # save mesh
    mesh = result[OutputKeys.OUTPUT]['mesh']
    write_obj(os.path.join(save_root, 'mesh.obj'), mesh)

    # save rendered color video
    frames_color = result[OutputKeys.OUTPUT]['frames_color']
    imageio.mimwrite(os.path.join(save_root, 'render_color.gif'), frames_color, duration=30)
    del frames_color

    # save rendered normals video
    frames_normals = result[OutputKeys.OUTPUT]['frames_normal']
    imageio.mimwrite(os.path.join(save_root, 'render_normals.gif'), frames_normals, duration=30)
    del frames_normals
    print(f'Output written to {os.path.abspath(save_root)}')


if __name__ == "__main__":
    human3d = pipeline(Tasks.human3d_render, model='damo/cv_3d-human-synthesis-library')
    input = {'dataset_id': 'damo/3DHuman_synthetic_dataset',  # 3dhuman-syn dataset, fixed
             'case_id': '000182',  # character id
             'resolution': 1280,  # render resolution
             }
    output = human3d(input)
    save_results(output, './human3d_results')
    print('download and render finished!')

    # # character ids of 3DHuman-Syn Dataset
    # url = 'https://modelscope.cn/api/v1/datasets/damo/3DHuman_synthetic_dataset/repo?Revision=master&FilePath=character_ids.txt'
    # charac_id_list = list_from_url(url)
    # print('character id list:', charac_id_list)








