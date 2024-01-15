import gradio as gr
import os
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.msdatasets import MsDataset
from modelscope import snapshot_download
import requests
from lib.animation import ms_wrapper
from functools import partial

# install blender
assets_dir = snapshot_download('damo/cv_3d-human-animation', cache_dir='.')
# assets_dir = 'damo/cv_3d-human-animation'
blender_file = os.path.join(assets_dir, '3D-assets', 'blender-3.1.2-linux-x64.tar.xz')
blender_path = os.path.join(assets_dir, '3D-assets', 'blender-3.1.2-linux-x64', 'blender')

if not os.path.exists(blender_file):
    raise Exception('found blender file failed.')
if not os.path.exists(blender_path):
    cmd = f'tar -xvf {blender_file} -C {os.path.join(assets_dir, "3D-assets")}'
    os.system(cmd)


CHARAC_TRIGGER_WORD = {
    'Curry': ['200000'],
    'SpiderMan': ['200002'],
    'IronMan': ['200004'],

    'Messi': ['200001'],
    'OprahWinfrey': ['200003'],
    'Woman': ['000182'],

    'BlondLady': ['000528'],
    'OldLady': ['000284'],
    'OldMan': ['000892'],

    '001116':[],
    '001162': [],
    '001079': [],

    '000762': [],
    '000466': [],
    '000477': [],

    '000243': [],
    '000456': [],
    '001118': [],

    '000697': [],
    '001001': [],
    '000259': [],

    '000710': [],
    '000902': [],
    '000820': [],

}

ACTION_TRIGGER_WORD = {
    'GoalkeeperCatch': [],
    'comb_goalkeeper': [],
    'SoccerSpin': [],
    'SoccerPass': [],

    'comb_run': [],
    'comb_superhero': [],
    # 'FlipKick': [],
    'Cartwheel': [],

    # 'SwingDancing': [],
    'ArmsHipHopDance': [],
    'GangnamStyle': [],
    # 'HipHopDancing': [],

    'ModernDancing': [],
    'StandardWalk': [],
    # 'WalkInCircle': [],
    'JogInCircle':[],

    'Dying':[],
    'AirSquat': [],
    'JabCross': [],
    'JumpingRope': [],
    # 'CatwalkWalkTurn180Tight': [],
}

class WebApp():
    def __init__(self, debug_mode=False):
        self.args_base = {
            "seed": 42,
            "device": "cuda",
            "output_dir": "output_demo",
            "inv_steps": 50,
            "spl_steps": 50,
            "img": None,
            "pos_prompt": '',
            "neg_prompt": 'worst quality, blurry, NSFW',
            "alpha": 3.0,
            "beta": 0.5,
            "omega": 15,
            "mask": None,
            "charac": "Curry",
            "charac_id": '',
            "charac_file": None,
            "charac_dir": "./assets/characters",
            "action_dir": "./assets/actions",
            "no_injection": False,
        }
        self.back_color = [0.858, 0.858, 0.858, 1.0]

        self.args_input = {}  # for gr.components only
        self.gr_charac = list(CHARAC_TRIGGER_WORD.keys())
        self.gr_action = list(ACTION_TRIGGER_WORD.keys())
        self.charac_id = 'Curry'

        # fun fact: google analytics doesn't work in this space currently
        self.gtag = os.environ.get('GTag')

        self.ga_script = f"""
            <script async src="https://www.googletagmanager.com/gtag/js?id={self.gtag}"></script>
            """
        self.ga_load = f"""
            function() {{
                window.dataLayer = window.dataLayer || [];
                function gtag(){{dataLayer.push(arguments);}}
                gtag('js', new Date());

                gtag('config', '{self.gtag}');
            }}
            """

        # # pre-download base model for better user experience
        # self._preload_pipeline()

        self.debug_mode = debug_mode  # turn off clip interrogator when debugging for faster building speed
        # if not self.debug_mode:
        #     self.init_interrogator()

        # clear cache
        data_dir = \
            MsDataset.load('3DHuman_synthetic_dataset', namespace='damo', subset_name='200000').config_kwargs[
                'split_config']['test']
        dataset_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(data_dir))))
        print(dataset_dir)
        # remove data_dir
        if os.path.exists(dataset_dir):
            os.system(f'rm -rf {dataset_dir}')


    def _preload_pipeline(self):
        pass

    def title(self):

        gr.HTML(
            """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <a href="https://menyifang.github.io/projects/En3D/index.html" style="margin-right: 20px; text-decoration: none; display: flex; align-items: center;">
            </a>
            <div>
                <h1 >\N{fire}En3D: Generative 3D Humans & Rigged Animation</h1>
                <h4 >v1.1</h4>
                <h5 style="margin: 0;">
                The latest updates will be available at our repository, <a href="https://github.com/menyifang/En3D" target="_blank">Github star it here</a>.
                </h5>

                <div style="display: flex; justify-content: center; align-items: center; text-align: center; margin: 20px; gap: 10px;">
                    <a class="flex-item" href="https://arxiv.org/abs/2401.01173" target="_blank">
                        <img src="https://img.shields.io/badge/Paper-arXiv-darkred.svg" alt="arXiv Paper">
                    </a>                      
                    <a class="flex-item" href="https://menyifang.github.io/projects/En3D/index.html" target="_blank">
                        <img src="https://img.shields.io/badge/Project_Page-En3D-green.svg" alt="Project Page">
                    </a>
                    <a class="flex-item" href="https://github.com/menyifang/En3D" target="_blank">
                        <img src="https://img.shields.io/badge/Github-Code-blue.svg" alt="GitHub Code">
                    </a>
                    
                </div>
            </div>
            </div>
            """
        )
        gr.HTML(
            """
            <div style="display: flex; justify-content: center; align-items: center; text-align: center;">
            <div>
            <h3>  
            With En3D, you can generate your own 3D avatars and animate them, all outputs are compatible with modern graphics engines.                              
            </h3>
            <h4 style="margin: 0;">
            v1.1 version built-in a <a href="https://github.com/menyifang/En3D" target="_blank">synthetic avatar library</a>
             (around 1000 avatars) and an
              <a href="https://github.com/menyifang/En3D" target="_blank">action library</a> (1000+ actions) for quick experience.
            </h4>

            </div>
            </div>
            """)

    def get_image(self):
        self.args_input['img'] = gr.Image(label='content image', type='pil', show_share_button=False,
                                          elem_classes="input_image")

    def get_prompts(self):
        generate_prompt = gr.Checkbox(label='generate prompt with clip', value=True)
        self.args_input['pos_prompt'] = gr.Textbox(label='prompt')

        # event listeners
        self.args_input['img'].upload(self._interrogate_image, inputs=[self.args_input['img'], generate_prompt],
                                      outputs=[self.args_input['pos_prompt']])
        generate_prompt.change(self._interrogate_image, inputs=[self.args_input['img'], generate_prompt],
                               outputs=[self.args_input['pos_prompt']])

    def _interrogate_image(self, image, generate_prompt):
        if hasattr(self, 'ci') and generate_prompt:
            return self.ci.interrogate_fast(image).split(',')[0].replace('arafed', '')
        else:
            return ''


    def get_charac_text(self):
        # self.args_input['charac_id'] = gr.Textbox(label='More character input (optional)',
        #                                             info='for more characters, you can also directly input “charac_id“ of avatars in library\n, stay it empty if you want to use above gallery.',
        #                                           )
        self.args_input['charac_id'] = gr.Textbox(label='charac_id input (optional)',
                                                  info='input the “charac_id“ of avatars in library for more models\n',
                                                  )
        # gr.HTML(
        #     """
        #     <div>
        #         <h6 style="margin: 0;">
        #             click
        #         <a href="https://modelscope.cn/datasets/damo/3DHuman_synthetic_dataset/summary" target="_blank">here</a>
        #         for avatar library.
        #         </h6>
        #     """
        #         )
    def get_charac_file(self):
        self.args_input['charac_file'] = gr.Files(label="Upload file (optional)", file_types=["file"], file_count='single')


    def get_charac(self, num_cols=3):
        self.args_input['charac'] = gr.State('Curry')
        lora_gallery = gr.Gallery(label='Input character', columns=num_cols, height=500,
                                  value=[(os.path.join(self.args_base['charac_dir'], f"{charac}.png"), charac) for charac in
                                         self.gr_charac], allow_preview=False, show_share_button=False, selected_index=0)
        lora_gallery.select(self._update_charac_selection, inputs=[], outputs=[self.args_input['charac']])

    def _update_charac_selection(self, selected_state: gr.SelectData):
        return self.gr_charac[selected_state.index]


    def get_action_text(self):
        self.args_input['action_id'] = gr.Textbox(label='action_id input (optional)',
                                                  info='input the “action_id“ of actions in library for more actions\n',
                                                  # info='for more actions, you can also directly input “action_id“ in action library\n, stay it empty if you want to use above gallery actions.',
                                                  )
        # self.args_input['action_id'] = gr.Textbox(label='More action input (optional)',
        #                                             info='for more actions, you can also directly input “action_id“ in action library\n, stay it empty if you want to use above gallery actions.',
        #                                           )
        # gr.HTML(
        #     """
        #     <div>
        #         <h6 style="margin: 0;">
        #             click <a href="https://modelscope.cn/datasets/damo/3DHuman_action_dataset/summary" target="_blank">here</a>
        #         for action library.
        #         </h6>
        #     """
        #         )

    def get_action_file(self):
        self.args_input['action_file'] = gr.Files(label="Upload file (optional)", file_types=["file"],
                                                  file_count='single')

    def get_action(self, num_cols=3):
        self.args_input['action'] = gr.State('GoalkeeperCatch')
        action_gallery = gr.Gallery(label='Input action', columns=num_cols, height=400,
                                  value=[(os.path.join(self.args_base['action_dir'], f"{action}.gif"), action) for action in
                                         self.gr_action], allow_preview=False, show_share_button=False, selected_index=0)
        action_gallery.select(self._update_action_selection, inputs=[], outputs=[self.args_input['action']])

    def _update_action_selection(self, selected_state: gr.SelectData):
        return self.gr_action[selected_state.index]


    def get_charac_id(self, case_name):
        if case_name in CHARAC_TRIGGER_WORD.keys():
            item = CHARAC_TRIGGER_WORD[case_name]
            if len(item) > 0:
                charac_id = item[0]
            else:
                charac_id = case_name
        else:
            charac_id = case_name

        return charac_id

    def extract_glb_from_zip(self, zip_file):
        # extract glb file from zip
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # extract zip to the same dir
            zip_ref.extractall(os.path.dirname(zip_file))
            # *.glb in zip
            glb_file = [f for f in zip_ref.namelist() if f.endswith('.glb')][0]
        return glb_file

    def extract_obj_from_zip(self, zip_file):
        # extract obj file from zip
        import zipfile
        with zipfile.ZipFile(zip_file, 'r') as zip_ref:
            # extract zip to the same dir
            zip_ref.extractall(os.path.dirname(zip_file))
            # *.glb in zip
            obj_file = [f for f in zip_ref.namelist() if f.endswith('.obj')][0]
        return obj_file


    def run_load3d(self, *values):
        print(values)
        gr_args = self.args_base.copy()
        print(self.args_input.keys())
        for k, v in zip(list(self.args_input.keys()), values):
            gr_args[k] = v
        charac_name = gr_args['charac']
        charac_id = self.get_charac_id(charac_name)

        charac_id_text = gr_args['charac_id']
        charac_id_file = gr_args['charac_file']
        print('charac_id', charac_id)
        print('charac_id_text', charac_id_text)
        print('charac_id_file', charac_id_file)
        use_file = False
        if charac_id_file is not None:
            charac_id = charac_id_file
            use_file = True
        elif charac_id_text != '':
            charac_id = charac_id_text

        if use_file:
            try:
                glb_path = self.extract_glb_from_zip(charac_id)
                glb_path = os.path.join(os.path.dirname(charac_id), glb_path)
            except:
                raise gr.Error(".glb not found in uploaded files, please check the upload.")

        else:
            try:
                data_dir = \
                    MsDataset.load('3DHuman_synthetic_dataset', namespace='damo', subset_name=charac_id).config_kwargs[
                        'split_config']['test']
                case_dir = os.path.join(data_dir, charac_id)
                glb_path = os.path.join(case_dir, 'body.glb')
            except:
                raise gr.Error("Character not found in library, please check the input.")



        return glb_path

    def run_animate(self, *values):
        gr_args = self.args_base.copy()
        print(self.args_input.keys())
        for k, v in zip(list(self.args_input.keys()), values):
            gr_args[k] = v
            print(k, v)

        charac_name = gr_args['charac']
        charac_id = self.get_charac_id(charac_name)

        charac_id_text = gr_args['charac_id']
        charac_id_file = gr_args['charac_file']
        if charac_id_file is not None:
            try:
                obj_path = self.extract_obj_from_zip(charac_id_file)
                obj_path = os.path.join(os.path.dirname(charac_id_file), obj_path)
                charac_id = obj_path
            except:
                raise gr.Error(".obj not found in uploaded avatar files, please check the upload.")
        elif charac_id_text != '':
            charac_id = charac_id_text
        print('charac_id', charac_id)

        action_id = gr_args['action']
        action_id_text = gr_args['action_id']
        action_id_file = gr_args['action_file']
        if action_id_file is not None:
            action_id = action_id_file
        elif action_id_text != '':
            action_id = action_id_text
        print('action_id: ', action_id)

        human3d = pipeline('human3d-animation-cus', model='damo/cv_3d-human-animation', revision='v1.0.0')

        input = {'dataset_id': 'damo/3DHuman_synthetic_dataset',
                 'case_id': charac_id,
                 'action_dataset': 'damo/3DHuman_action_dataset',
                 'action': action_id,
                 'save_dir': 'human3d_results',
                 'blender': blender_path}
        output = human3d(input)
        result = output['output']
        return result


    def preset_library(self):
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    self.get_charac(num_cols=3)
                    # self.get_charac_text()
                    with gr.Row():
                        self.get_charac_text()
                        self.get_charac_file()
                    submit_btn_load3d = gr.Button("Load 3D Avatar!", variant='primary')

                res_charac = gr.Model3D(clear_color=self.back_color, camera_position=[90, 90, 4],
                                        label="3D Model Result", interactive=False)

            submit_btn_load3d.click(self.run_load3d,
                             inputs=list(self.args_input.values()),
                             outputs=[res_charac],
                             scroll_to_output=True,
                             )

            with gr.Row():
                with gr.Column():
                    self.get_action(num_cols=4)
                    # self.get_action_text()
                    with gr.Row():
                        self.get_action_text()
                        self.get_action_file()
                    submit_btn_animate = gr.Button("Animate it!", variant='primary')

                res_animation = gr.Model3D(clear_color=self.back_color, camera_position=[90, 90, 5],
                                            label="Animated Avatar Result")

            submit_btn_animate.click(self.run_animate,
                                        inputs=list(self.args_input.values()),
                                        outputs=[res_animation],
                                        scroll_to_output=True,
                                        )

            gr.Examples(examples=[
                ['Curry', 'GoalkeeperCatch'],
                ['SpiderMan', 'comb_superhero'],
            ], inputs=[self.args_input['charac_id'], self.args_input['action_id']],
                examples_per_page=20, label="示例", elem_id="examples"
            )

    def ui(self):
        with gr.Blocks() as demo:
            self.title()
            with gr.Tabs():
                with gr.TabItem('Animate Synthetic Avatars'):
                    self.preset_library()
                with gr.TabItem('Text-to-3D Avatar(Coming soon)'):
                    gr.Markdown("## <center>Coming soon!</center>")
                with gr.TabItem('Image-to-3D Avatar(Coming soon)'):
                    gr.Markdown("## <center>Coming soon!</center>")

            demo.load(None, js=self.ga_load)
            # demo.load()

        return demo

app = WebApp(debug_mode=False)
demo = app.ui()

if __name__ == "__main__":
    demo.launch(server_name='30.24.152.138', server_port=8788)
    # demo.queue(max_size=100)
    # demo.launch(share=False)
