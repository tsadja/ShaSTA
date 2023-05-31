# Visualizing results
We are offering code for presenting interpretable, visually appealing qualitative results. We offer 2 methods for visualizing tracking results both from the top-down LiDAR view (basically BEV) with the LiDAR point cloud projected onto a road map underlay and from the front facing camera view with 3D bounding boxes projected onto the 2D image, so audiences can interpret our results with ease.

```bash
python nusc_visualize/visualize.py --raw_data_folder "data/nuScenes" --save_path "work_dir_visualize" \
--scene_name "scene-NUMBER" --render_class "OBJECT_TYPE" \
--track_result_path "results/SPLIT_tracking_result.json" --split "SPLIT"
```