from mmpose.apis import inference_topdown, init_model
from mmpose.utils import register_all_modules
import cv2
from mmcv.image import imread
from mmpose.registry import VISUALIZERS

radius = 3
kpt_thr = 0.3
draw_heatmap = True
thickness = 1
alpha = 0.8
skeleton_style = 'mmpose'
cfg_options = dict(model=dict(test_cfg=dict(output_heatmaps=True)))

register_all_modules()

config_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192.py'
checkpoint_file = 'td-hm_hrnet-w48_8xb32-210e_coco-256x192-0e67c616_20220913.pth'

for i in range(0, 403):
    img_path = "tests/crop_results_small/" + str(i) +".png"

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