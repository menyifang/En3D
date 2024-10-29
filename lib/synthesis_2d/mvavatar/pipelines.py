from PIL import Image
import os
import torch
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import random
from typing import Optional
import numpy as np

def extract_part_img(img, n_view):
    h, w, c = img.shape
    w = int(w / 7)
    front = img[:, :w, :]
    back = img[:, -w:, :]
    side = img[:, int(w*3): int(w*4),:]
    if n_view==3:
        concat = np.concatenate((front, back, side), axis=1)
    else:
        concat = np.concatenate((front, back), axis=1)
    return concat

class MVAvatar(torch.nn.Module):
    def __init__(self):
        super().__init__()

    @classmethod
    def get_random_description(
        self,
        description_type: str = 'base_description',
        seed: int = -1,
    ):
        if seed > 65535 or seed < 0:
            seed = random.randint(0, 65535)

        random.seed(seed)
        description = self.wildcards[description_type]
        num = len(description)
        random_seq = random.sample(range(0, num), 1)
        random_seq = random_seq[0]
        if random_seq == num:
            return ''
        else:
            return description[random_seq]

    @classmethod
    def to_device(self, device):
        self.pipe.to(device)

    @classmethod
    def from_pretrained(
            self,
            model_path: str = None,
            device: str = "cuda",
            n_view: int = 2,
    ):
        controlnet_pose_path = os.path.join(model_path, 'controlnet-pose')
        # controlnet_pose_path = '/data/qingyao/diffusion/code/diffusers/models/control_v11f1e_sd15_openpose'  # better

        controlnet_edge_path = os.path.join(model_path, 'controlnet-lineart')
        base_model_path = os.path.join(model_path, 'base-model')
        # base_model_path = '/data/qingyao/diffusion/code/diffusers/models/stable-diffusion-v1-5'

        controlnet_pose = ControlNetModel.from_pretrained(
            controlnet_pose_path,
            torch_dtype=torch.float16
        )

        controlnet_edge = ControlNetModel.from_pretrained(
            controlnet_edge_path,
            torch_dtype=torch.float16
        )

        self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
            base_model_path,
            controlnet=[controlnet_pose, controlnet_edge],
        ).to(torch_dtype=torch.float16, torch_device=device)

        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        self.pipe.enable_xformers_memory_efficient_attention()
        # self.pipe.vae.enable_tiling()

        pose_img = Image.open(os.path.join(model_path, 'control-imgs/pose.png')).convert('RGB')
        edge_img = Image.open(os.path.join(model_path, 'control-imgs/edge.png')).convert('RGB')

        if n_view == 7:
            self.pose = pose_img
            self.edge = edge_img
        else:
            # crop pose and edge
            pose_img = extract_part_img(np.array(pose_img), n_view=n_view)
            edge_img = extract_part_img(np.array(edge_img), n_view=n_view)
            self.pose = Image.fromarray(pose_img)
            self.edge = Image.fromarray(edge_img)
            # save pose
            self.pose.save(os.path.join('pose.png'))
            self.edge.save(os.path.join('edge.png'))



        base = open(os.path.join(model_path, 'wildcards/base.txt'), 'r')
        base_description = base.readlines()
        base_description = [x[:-1] for x in base_description]

        special_costume = open(os.path.join(model_path, 'wildcards/special_costume.txt'), 'r', encoding="utf-8")
        special_costume_description = special_costume.readlines()
        special_costume_description = [x[:-1] for x in special_costume_description]

        color = open(os.path.join(model_path, 'wildcards/color.txt'), 'r', encoding="utf-8")
        color_description = color.readlines()
        color_description = [x[:-1] for x in color_description]

        race = open(os.path.join(model_path, 'wildcards/race.txt'), 'r', encoding="utf-8")
        race_description = race.readlines()
        race_description = [x[:-1] for x in race_description]

        style = open(os.path.join(model_path, 'wildcards/style.txt'), 'r', encoding="utf-8")
        style_description = style.readlines()
        style_description = [x[:-1] for x in style_description]

        other = open(os.path.join(model_path, 'wildcards/other.txt'), 'r', encoding="utf-8")
        other_description = other.readlines()
        other_description = [x[:-1] for x in other_description]

        self.wildcards = {
            'base_description': base_description,
            'special_costume_description': special_costume_description,
            'color_description': color_description,
            'race_description': race_description,
            'style_description': style_description,
            'other_description': other_description,

        }
        return self

    @classmethod
    def inference(
            self,
            prompt: str = None,
            width: Optional[int] = 1152, #*2
            height: Optional[int] = 864,
            sample_steps: Optional[int] = 20,
            n_prompt: Optional[str] = None,
            seed: Optional[int] = -1,
    ):
        print(prompt)

        if prompt is None:

            prompt = 'three views drawing, ' + \
                           self.get_random_description('base_description', seed=seed) + \
                           self.get_random_description('special_costume_description', seed=seed) + \
                           self.get_random_description('other_description', seed=seed)
                           # + "wearing shoes, " + \
                           # 'white background, simple background, best quality, high resolution'
        else:
            prompt = f'three views drawing, {prompt}'
            prompt = prompt.replace('__base__',
                                    self.get_random_description('base_description', seed=seed))
            prompt = prompt.replace('__color__',
                                    self.get_random_description('color_description', seed=seed))
            prompt = prompt.replace('__race__',
                                    self.get_random_description('race_description', seed=seed))
            prompt = prompt.replace('__style__',
                                    self.get_random_description('style_description', seed=seed))
            prompt = prompt.replace('__other__',
                                    self.get_random_description('other_description', seed=seed))
            prompt = prompt.replace('__special_costume__',
                                    self.get_random_description('special_costume_description', seed=seed))

        print(prompt)
        print('seed: ', seed)

        if n_prompt is None:
            # n_prompt = "blur, haze, dark, dim, naked, nude, deformed iris, deformed pupils, semi-realistic, " \
            #            "mutated hands and fingers, deformed, distorted, disfigured, poorly drawn, bad anatomy, " \
            #            "wrong anatomy, extra limb, missing limb, floating limbs, disconnected limbs, mutation, " \
            #            "mutated, ugly, disgusting, amputation"
            n_prompt = "(mid-calf socks), (naked), (barefooted), paintings, sketches, fingers, (worst quality:2), (low quality:2), (normal quality:2), lowres, normal quality, ((monochrome)), ((grayscale)), skin spots, acnes, skin blemishes, age spot, (outdoor:1.6), backlight,(ugly:1.331), (duplicate:1.331), (morbid:1.21), (mutilated:1.21), (tranny:1.331), mutated hands, (poorly drawn hands:1.5), blurry, (bad anatomy:1.21), (bad proportions:1.331), extra limbs, (disfigured:1.331), (more than 2 nipples:1.331), (missing arms:1.331), (extra legs:1.331), (fused fingers:1.61051), (too many fingers:1.61051), (unclear eyes:1.331), lowers, bad hands, missing fingers, extra digit, (futa:1.1),bad hands, missing fingers, bad-hands-5"

        if seed < 0 or seed > 65535:
            seed = torch.seed()
            print(seed)
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)
        else:
            generator = torch.Generator(device=self.pipe.device).manual_seed(seed)

        image = self.pipe(
            controlnet_conditioning_scale=[1.0, .7],
            prompt=prompt,
            height=height,
            width=width,
            num_inference_steps=sample_steps,
            negative_prompt=n_prompt,
            generator=generator,
            image=[self.pose, self.edge],
        ).images[0]



        return image
