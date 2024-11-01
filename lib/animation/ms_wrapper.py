# Copyright (c) Alibaba, Inc. and its affiliates.
from modelscope.models.base import TorchModel
from modelscope.preprocessors.base import Preprocessor
from modelscope.pipelines.base import Model, Pipeline
from modelscope.utils.config import Config
from modelscope.pipelines.builder import PIPELINES
from modelscope.preprocessors.builder import PREPROCESSORS
from modelscope.models.builder import MODELS
from typing import Any, Dict
from modelscope.msdatasets import MsDataset
from modelscope.outputs import OutputKeys
from .generate_skeleton import gen_skeleton_bvh
from .utils import read_obj, write_obj
import os
import cv2
import logging
logger = logging.getLogger(__name__)

@PIPELINES.register_module('human3d-animation-cus', module_name='human3d-animation')
class MyCustomPipeline(Pipeline):
    """ Give simple introduction to this pipeline.

    Examples:

    >>> from modelscope.pipelines import pipeline
    >>> input = "Hello, ModelScope!"
    >>> my_pipeline = pipeline('my-task', 'my-model-id')
    >>> result = my_pipeline(input)

    """

    def __init__(self, model, device='gpu', **kwargs):
        """
        use model to create a image sky change pipeline for image editing
        Args:
            model (str or Model): model_id on modelscope hub
            device (str): only support gpu
        """
        super().__init__(model=model, **kwargs)
        self.model_dir = model
        print('model_dir:', self.model_dir)
        logger.info('model_dir:', self.model_dir)

    def preprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs

    def gen_skeleton(self, case_dir, action_dir, action):
        self.case_dir = case_dir
        self.action_dir = action_dir
        self.action = action
        status = gen_skeleton_bvh(self.model_dir, self.action_dir,
                                  self.case_dir, self.action)
        return status

    def gen_weights(self, save_dir=None):
        case_name = os.path.basename(self.case_dir)
        action_name = os.path.basename(self.action).replace('.npy', '')
        if save_dir is None:
            gltf_path = os.path.join(self.case_dir, '%s-%s.glb' %
                                     (case_name, action_name))
        else:
            os.makedirs(save_dir, exist_ok=True)
            gltf_path = os.path.join(save_dir, '%s-%s.glb' %
                                     (case_name, action_name))

        # current file directory name
        exec_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'skinning.py')

        cmd = f'{self.blender} -b -P {exec_path}  -- --input {self.case_dir}' \
              f' --gltf_path {gltf_path} --action {self.action}'
        print(cmd)
        os.system(cmd)
        return gltf_path

    def animate(self, mesh_path, action_dir, action, save_dir=None):
        case_dir = os.path.dirname(os.path.abspath(mesh_path))
        tex_path = mesh_path.replace('.obj', '.png')
        mesh = read_obj(mesh_path)
        tex = cv2.imread(tex_path)
        vertices = mesh['vertices']
        mesh['vertices'] = vertices
        mesh['texture_map'] = tex
        write_obj(mesh_path, mesh)

        self.gen_skeleton(case_dir, action_dir, action)
        gltf_path = self.gen_weights(save_dir)
        if os.path.exists(gltf_path):
            logger.info('save animation succeed!')
        else:
            logger.info('save animation failed!')
        return gltf_path

    def forward(self, input: Dict[str, Any]) -> Dict[str, Any]:
        dataset_id = input['dataset_id']
        case_id = input['case_id']
        action_data_id = input['action_dataset']
        action = input['action']
        if 'save_dir' in input:
            save_dir = input['save_dir']
        else:
            save_dir = None

        if 'blender' in input:
            self.blender = input['blender']
        else:
            self.blender = 'blender'

        if case_id.endswith('.obj'):
            mesh_path = case_id
        else:
            dataset_name = dataset_id.split('/')[-1]
            user_name = dataset_id.split('/')[0]
            data_dir = MsDataset.load(
                dataset_name, namespace=user_name,
                subset_name=case_id).config_kwargs['split_config']['test']
            case_dir = os.path.join(data_dir, case_id)
            mesh_path = os.path.join(case_dir, 'body.obj')
        logger.info('load mesh:', mesh_path)

        dataset_name = action_data_id.split('/')[-1]
        user_name = action_data_id.split('/')[0]
        action_dir = MsDataset.load(
            dataset_name, namespace=user_name,
            split='test').config_kwargs['split_config']['test']
        action_dir = os.path.join(action_dir, 'actions_a')

        output = self.animate(mesh_path, action_dir, action, save_dir)

        return {OutputKeys.OUTPUT: output}

    def postprocess(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        return inputs


