from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope import snapshot_download
import os
import requests

model_dir = snapshot_download('damo/cv_3d-human-animation', cache_dir='.')
# install blender
blender_file = os.path.join(model_dir, '3D-assets', 'blender-3.1.2-linux-x64.tar.xz')
blender_path = os.path.join(model_dir, '3D-assets', 'blender-3.1.2-linux-x64', 'blender')
if not os.path.exists(blender_file):
    raise Exception('found blender file failed.')
if not os.path.exists(blender_path):
    cmd = f'tar -xvf {blender_file} -C {os.path.join(model_dir, "3D-assets")}'
    os.system(cmd)

def list_from_url(url):
    r = requests.get(url)
    list = r.content.decode('utf-8').split('\n')
    list = list[:-1]
    return list


if __name__ == "__main__":
    human3d = pipeline(Tasks.human3d_animation, model='damo/cv_3d-human-animation')
    input = {'dataset_id': 'damo/3DHuman_synthetic_dataset', # character dataset, fixed
             'case_id': '000182', # character id, choose one from character dataset
             'action_dataset': 'damo/3DHuman_action_dataset', # action dataset, fixed
             'action': 'SwingDancing', # action id, choose one from action dataset
             'save_dir': 'human3d_results',  # save directory,
             'blender': blender_path, # blender path
             }
    output = human3d(input)
    print('saved animation file to %s' % output)
    print('finished!')

    # # character ids of 3DHuman-Syn Dataset
    # url = 'https://modelscope.cn/api/v1/datasets/damo/3DHuman_synthetic_dataset/repo?Revision=master&FilePath=character_ids.txt'
    # charac_id_list = list_from_url(url)
    # print('character id list:', charac_id_list)
    #
    # # action ids of 3DHuman-Action Dataset
    # url = 'https://modelscope.cn/api/v1/datasets/damo/3DHuman_action_dataset/repo?Revision=master&FilePath=action_ids.txt'
    # action_id_list = list_from_url(url)
    # print('action id list:', action_id_list)






