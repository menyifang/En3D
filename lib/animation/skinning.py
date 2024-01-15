# Copyright (c) Alibaba, Inc. and its affiliates.

import bpy
import os
import sys
import argparse
from mathutils import Matrix


class ArgumentParserForBlender(argparse.ArgumentParser):
    """
    This class is identical to its superclass, except for the parse_args
    method (see docstring). It resolves the ambiguity generated when calling
    Blender from the CLI with a python script, and both Blender and the script
    have arguments. E.g., the following call will make Blender crash because
    it will try to process the script's -a and -b flags:
    >>> blender --python my_script.py -a 1 -b 2

    To bypass this issue this class uses the fact that Blender will ignore all
    arguments given after a double-dash ('--'). The approach is that all
    arguments before '--' go to Blender, arguments after go to the script.
    The following calls work fine:
    >>> blender --python my_script.py -- -a 1 -b 2
    >>> blender --python my_script.py --
    """

    def _get_argv_after_doubledash(self):
        """
        Given the sys.argv as a list of strings, this method returns the
        sublist right after the '--' element (if present, otherwise returns
        an empty list).
        """
        try:
            idx = sys.argv.index("--")
            return sys.argv[idx + 1:]  # the list after '--'
        except ValueError as e:  # '--' not in the list:
            return []

    # overrides superclass
    def parse_args(self):
        """
        This method is expected to behave identically as in the superclass,
        except that the sys.argv list will be pre-processed using
        _get_argv_after_doubledash before. See the docstring of the class for
        usage examples and details.
        """
        return super().parse_args(args=self._get_argv_after_doubledash())


def getKeyframes(ob):
    if ob.type in ['MESH', 'ARMATURE'] and ob.animation_data:
        for fc in ob.animation_data.action.fcurves:
            if fc.data_path.endswith('rotation_euler'):

                keyframe_list = []
                for key in fc.keyframe_points:
                    # print('frame:',key.co[0],'value:',key.co[1])
                    keyframe_list.append(key.co[0])

                keyframe_list = list(set(keyframe_list))
                print('keyframe_list:')
                # print(keyframe_list)
                print(len(keyframe_list))
                firstKFN = int(keyframe_list[0])
                lastKFN = int(keyframe_list[-1])
                # Only needs to check animation of one bone
                return firstKFN, lastKFN


def init_scene():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)


def add_material_for_obj(obj, filepath):
    obj.data.materials.clear()
    # Load image into Blender
    mat_name = 'mat' + '_%s' % os.path.basename(filepath)[:-4]
    mat = bpy.data.materials.new(name=mat_name)
    mat.use_nodes = True
    obj.data.materials.append(mat)
    matnodes = mat.node_tree.nodes
    tex = matnodes.new("ShaderNodeTexImage")
    # Assign the loaded image to the diffuse texture node
    tex.image = bpy.data.images.load(filepath)
    disp = bpy.data.materials[mat_name].node_tree.nodes["Principled BSDF"].inputs['Base Color']
    mat.node_tree.links.new(disp, tex.outputs[0])


def import_obj(obj_path, img_path=None):
    bpy.ops.import_scene.obj(filepath=obj_path, split_mode="OFF")
    bpy.ops.object.shade_smooth()
    # For some mysterious raison, this is necessary otherwise I cannot toggle shade smooth / shade flat
    mesh = bpy.context.selected_objects[0]
    if img_path is not None and os.path.exists(img_path):
        add_material_for_obj(mesh, img_path)
    else:
        print('no texture for %s' % obj_path)
    return mesh


def import_skeleton(filepath):
    bpy.ops.import_anim.bvh(filepath=filepath, filter_glob="*.bvh", target='ARMATURE', global_scale=1, frame_start=1,
                            use_fps_scale=False, use_cyclic=False, rotate_mode='NATIVE', axis_forward='-Z', axis_up='Y')


def export_animated_mesh(gltf_path, IsAnimation):
    # Create output directory if needed
    output_dir = os.path.dirname(os.path.abspath(gltf_path))
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    bpy.ops.object.select_all(action='SELECT')

    if gltf_path != '':
        bpy.ops.export_scene.gltf(filepath=gltf_path, export_format='GLB', export_selected=True,
                                  export_morph=IsAnimation)


def remove_keyframes(object, frame):
    action = object.animation_data.action
    if action is None:
        return
    for fc in action.fcurves:
        object.keyframe_delete(data_path=fc.data_path, frame=frame)


def apply_transfrom(ob, use_location=False, use_rotation=False, use_scale=False):
    mb = ob.matrix_basis
    I = Matrix()
    loc, rot, scale = mb.decompose()

    # rotation
    T = Matrix.Translation(loc)
    R = mb.to_3x3().normalized().to_4x4()
    S = Matrix.Diagonal(scale).to_4x4()

    transform = [I, I, I]
    basis = [T, R, S]

    def swap(i):
        transform[i], basis[i] = basis[i], transform[i]

    if use_location:
        swap(0)
    if use_rotation:
        swap(1)
    if use_scale:
        swap(2)

    M = transform[0] @ transform[1] @ transform[2]
    if hasattr(ob.data, "transform"):
        ob.data.transform(M)
    for c in ob.children:
        c.matrix_local = M @ c.matrix_local

    ob.matrix_basis = basis[0] @ basis[1] @ basis[2]


if __name__ == '__main__':
    parser = ArgumentParserForBlender()
    parser.add_argument('--input', dest='input_dir', type=str, required=True,
                        help='Input directory')
    parser.add_argument('--gltf_path', dest='gltf_path', type=str, required=True,
                        help='Input directory')
    parser.add_argument('--action', dest='action', type=str, required=False,
                        help='action name')

    args = parser.parse_args()

    input_dir = args.input_dir
    gltf_path = args.gltf_path
    action = args.action

    init_scene()

    obj_path = os.path.join(input_dir, 'body.obj')
    img_path = os.path.join(input_dir, 'body.png')
    mesh = import_obj(obj_path, img_path=img_path)

    skeleton_path = os.path.join(input_dir, 'skeleton_a.bvh')
    import_skeleton(skeleton_path)
    skeleton = bpy.context.selected_objects[0]
    bpy.context.scene.render.fps = 30

    ## resize mesh, ske
    times = 10000
    mesh.scale = (times, times, times)
    skeleton.scale = (times, times, times)

    mesh.select_set(False)
    skeleton.select_set(True)
    bpy.context.view_layer.objects.active = skeleton

    bpy.ops.object.mode_set(mode='POSE')
    bpy.ops.pose.transforms_clear()

    bpy.ops.object.mode_set(mode='OBJECT')

    mesh.select_set(True)
    skeleton.select_set(True)
    bpy.context.view_layer.objects.active = skeleton

    bpy.ops.object.parent_set(type='ARMATURE_AUTO')

    ob = bpy.context.active_object
    ob.scale = (1, 1, 1)
    apply_transfrom(mesh, use_scale=True)

    remove_keyframes(bpy.context.object, 1)

    firstKFN, lastKFN = getKeyframes(skeleton)
    IsAnimation = not (firstKFN == lastKFN)
    if IsAnimation:
        # Set Frame start and end
        bpy.data.scenes[0].frame_start = firstKFN
        bpy.data.scenes[0].frame_end = lastKFN
        print("This is Animated Model")
    else:
        print("This is Static Model")

    skeleton.name = "Armature"
    mesh.name = "body"

    export_animated_mesh(gltf_path, IsAnimation=False)
