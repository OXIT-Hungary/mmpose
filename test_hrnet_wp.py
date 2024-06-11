from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import cv2
import os
import shutil
from mmcv.image import imread
from mmpose.registry import VISUALIZERS

def get_image_list(path):
    img_list = []
    for file in os.listdir(path):
        name, ext = os.path.splitext(file)
        if ext == '.jpg' or ext == '.png':
            img_list.append(file)
    return img_list

def main(cfg):

    radius = cfg.radius
    kpt_thr = cfg.kpt_thr
    draw_heatmap = cfg.draw_heatmap
    thickness = cfg.thickness
    alpha = cfg.alpha
    skeleton_style = cfg.skeleton_style
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

    register_all_modules()

    config_file = cfg.config_file
    checkpoint_file = cfg.checkpoint_file

    if os.path.isdir(cfg.output_path):
        shutil.rmtree(cfg.output_path)
        os.makedirs(cfg.output_path, exist_ok=True)
    else:
        os.makedirs(cfg.output_path, exist_ok=True)

    img_list = get_image_list(cfg.img_path)

    for img_name in img_list:

        img = cfg.img_path + img_name

        img = imread(img, channel_order='rgb')
        img = cv2.resize(img,(cfg.img_size_w, cfg.img_size_h))

        model = init_model(config_file, checkpoint_file, device='cuda:0', cfg_options=cfg_options)  # or device='cuda:0'

        # please prepare an image with person
        results = inference_topdown(model, img)
        print(type(results[0]))
        print(len(results))
        #cv2.imshow("test", results)

        # init visualizer
        model.cfg.visualizer.radius = radius
        model.cfg.visualizer.alpha = alpha
        model.cfg.visualizer.line_width = thickness

        visualizer = VISUALIZERS.build(model.cfg.visualizer)
        visualizer.set_dataset_meta(
            model.dataset_meta, skeleton_style=skeleton_style)

        visualizer.add_datasample(
            'result',
            img,
            data_sample=results[0],
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=kpt_thr,
            draw_heatmap=draw_heatmap,
            skeleton_style=skeleton_style,
            out_file=cfg.output_path + img_name)