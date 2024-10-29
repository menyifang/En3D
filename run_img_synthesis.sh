#!/bin/bash

CASE=test
IMG=assets/${CASE}.png

WORK_DIR=$(pwd)
SAVE_DIR=${WORK_DIR}/res_img
OUT=${SAVE_DIR}/${CASE}
outdir=${OUT}/PTI_render

# Generate multi-views && coarse mesh
MODEL=models/model_human.pkl
python lib/3DGEN/projector_img.py --target=${IMG} --outdir=${OUT} --idx 0 --network $MODEL --camera=models/camera.json

# Generate corresponding normals & masks
SEED=seed0000
DATA_DIR=${outdir}_multiview/${SEED}
IMG_DIR=${DATA_DIR}/image
NORM_DIR=${DATA_DIR}/normal
WORK_DIR=$(pwd)

## generate normals using ICON, install and download models following https://github.com/YuliangXiu/ICON
cd [your_ICON_directory]
#[optional] source activate icon_env  #setup extra conda env if conflict exists
python -m apps.infer_normal_fixpose -cfg ./configs/icon-filter.yaml \
-gpu 0 -in_dir $IMG_DIR -out_dir $NORM_DIR \
-export_video -loop_smpl 100 -loop_cloth 200 -hps_type pixie
cd ${WORK_DIR}

## generate masks
python lib/3DGEN/gen_mask.py --data_dir $DATA_DIR

# Geometry refinement
cd ${WORK_DIR}/lib/opt_geo
python opt_3dgan.py --config 3d_gen.json --data_dir ${DATA_DIR} --out_dir ${DATA_DIR}

MESH=${DATA_DIR}/dmtet_mesh/mesh.obj
SMPL=${WORK_DIR}/models/smpl_center.obj
python refine/refine_body.py --mesh_path $MESH --smpl_obj_path $SMPL --operation "refine_arm"
cd ${WORK_DIR}

# Texture modeling
## generate UV texture
cd ${WORK_DIR}/lib/opt_tex
REFINE=${DATA_DIR}/dmtet_mesh/dmtet_mesh_refine.obj
python gen_texture.py --tex_size 1024 --smpl_obj_path $SMPL --refine_path $REFINE --data_dir ${DATA_DIR} --use_idx '0'

## refine UV texture
TEX=${DATA_DIR}/dmtet_mesh/dmtet_mesh_uv_tex.png
TEX_OUT=${DATA_DIR}/dmtet_mesh/dmtet_mesh_refine_tex.png
python refine_texture.py --inpath $TEX --outpath $TEX_OUT


# Rendering
FINAL=${DATA_DIR}/dmtet_mesh/dmtet_mesh_refine_tex.obj
python render_texture.py --final_path $FINAL

cd ${WORK_DIR}

