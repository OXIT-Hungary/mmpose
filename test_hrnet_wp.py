from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import cv2
from mmcv.image import imread
from mmpose.registry import VISUALIZERS

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

    for i in range(0, 403):
        img_path = cfg.img_path + str(i) + ".png"

        model = init_model(config_file, checkpoint_file, device='cuda:0', cfg_options=cfg_options)  # or device='cuda:0'

        # please prepare an image with person
        results = inference_topdown(model, img_path)
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


        img = imread(img_path, channel_order='rgb')
        visualizer.add_datasample(
            'result',
            img,
            data_sample=results[0],
            draw_gt=False,
            draw_bbox=True,
            kpt_thr=kpt_thr,
            draw_heatmap=draw_heatmap,
            skeleton_style=skeleton_style,
            out_file="output/"+ str(i) + " .png")