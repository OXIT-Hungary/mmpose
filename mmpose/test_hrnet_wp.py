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

def main_init(cfg):
    dataloader = cfg.models.mmpose.dataloader

    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))
    config_file = dataloader.config_file
    checkpoint_file = dataloader.checkpoint_file

    model = init_model(config_file, checkpoint_file, device='cuda:0', cfg_options=cfg_options)  # or device='cuda:0'

    return model

def main_eval(cfg, model, img, bboxes):

    outputs = []

    for idx, bbox in enumerate(bboxes):

        if all(i >= 0 for i in bbox):
            cropped_img = img[int(bbox[1]):int(bbox[3]), int(bbox[0]):int(bbox[2])]

            parameters = cfg.models.mmpose.parameters
            datawriter = cfg.models.mmpose.datawriter

            radius = parameters.radius
            kpt_thr = parameters.kpt_thr
            draw_heatmap = parameters.draw_heatmap
            thickness = parameters.thickness
            alpha = parameters.alpha
            skeleton_style = parameters.skeleton_style
            
            register_all_modules()

            _img = cv2.resize(cropped_img,(parameters.img_size_w, parameters.img_size_h))

            # please prepare an image with person
            results = inference_topdown(model, _img)
            #print(type(results[0]))
            #print(len(results))
            #print('-Pose estimated-')

            outputs.append(results)
        else:
            outputs.append([])

    return outputs

def main(cfg):

    parameters = cfg.models.mmpose.parameters
    dataloader = cfg.models.mmpose.dataloader
    datawriter = cfg.models.mmpose.datawriter

    radius = parameters.radius
    kpt_thr = parameters.kpt_thr
    draw_heatmap = parameters.draw_heatmap
    thickness = parameters.thickness
    alpha = parameters.alpha
    skeleton_style = parameters.skeleton_style
    cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

    register_all_modules()

    config_file = dataloader.config_file
    checkpoint_file = dataloader.checkpoint_file

    if os.path.isdir(datawriter.output_path):
        shutil.rmtree(datawriter.output_path)
        os.makedirs(datawriter.output_path, exist_ok=True)
    else:
        os.makedirs(datawriter.output_path, exist_ok=True)

    img_list = get_image_list(dataloader.img_path)

    for img_name in img_list:

        img = dataloader.img_path + img_name

        img = imread(img, channel_order='rgb')
        img = cv2.resize(img,(parameters.img_size_w, parameters.img_size_h))

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
            out_file=datawriter.output_path + img_name)