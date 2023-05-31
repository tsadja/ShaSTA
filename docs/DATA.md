# Get nuScenes dataset and detector information
This is how the nuScenes dataset and pre-processed data should be organized in the data folder.

```
# For nuScenes Dataset         
└── data
     ├── nuScenes
          ├── samples       <-- key frames
          ├── sweeps        <-- frames without annotation
          ├── maps          <-- unused
          ├── v1.0-trainval <-- metadata for training and validation splits
          ├── v1.0-test     <-- main test folder 
               ├── samples       <-- key frames
               ├── sweeps        <-- frames without annotation
               ├── maps          <-- unused
               |── v1.0-test     <-- metadata and annotations
     ├── nusc_preprocessed
          ├── bev_map.pth   <-- pre-trained LiDAR backbone
          ├── cp
               ├── train.json    <-- training detections
               ├── test.json     <-- testing detections
               ├── val.json      <-- validation detections
          ├── train_2hz       <-- training data
               ├── detections    <-- preprocessed detections per detector used
               ├── ego_info      <-- ego pose information
               ├── gt_info       <-- nuScenes ground-truth information
               ├── gt_shasta     <-- ShaSTA ground-truth affinity matrix per detector used
               ├── token_info    <-- tokens information for each scene
          ├── val_2hz         <-- validation data
               ├── detections    <-- preprocessed detections per detector used
               ├── ego_info      <-- ego pose information
               ├── token_info    <-- tokens information for each scene
          ├── test_2hz        <-- test data
               ├── detections    <-- preprocessed detections per detector used
               ├── ego_info      <-- ego pose information
               ├── token_info    <-- tokens information for each scene
          ├── test_frame_info.json   <-- frame tokens in test data
          ├── train_frame_info.json  <-- frame tokens in train data
          ├── val_frame_info.json    <-- frame tokens in val data
          ├── infos_test_10sweeps_withvelo.pkl                <-- LiDAR with 10 sweeps for test data (in the style of CenterPoint and mmdet)
          ├── infos_train_10sweeps_withvelo_filter_True.pkl   <-- LiDAR with 10 sweeps for train data 
          ├── infos_val_10sweeps_withvelo_filter_True.pkl     <-- LiDAR with 10 sweeps for validation data 

```

## Download nuScenes data

All of the content in nuScenes directory can be obtained from the [nuScenes download website](https://www.nuscenes.org/nuscenes#download). We understand that getting this data onto a machine may not be immediately obvious. Our recommendation is to begin downloading the data to your local machine in order to get the AWS link that it is coming from in your downloads page. Then, cancel the local download job and using the AWS link, go to your cluster where you actually want this data and run:
```bash
wget AWS_LINK
```
You will need to run this 10 times, since the creator of nuScenes have split the trainval data into 10 separate downloads. For the test data, you only need one download. 

## Preprocess data

### Download zip file with preprocessed data (fast)

Next, for the nusc_preprocssed directory, we recommend downloading the data from the zip file we provide as follows:
```bash
wget http://download.cs.stanford.edu/juno/ShaSTA/nusc_preprocessed.zip
unzip nusc_preprocessed.zip
```

### Generate preprocessed data yourself (time-consuming)

If you want to generate the preprocessed data yourself (we do not recommend since it is time-consuming), please follow the CenterPoint instructions for getting the infos pkl files and for the remainder that is ShaSTA-specific, please run:
```bash
chmod +x preprocessing.sh
./preprocessing.sh
```

If you want to run this algorithm with other 3D detections, i.e. not CenterPoint, then we recommend using the preprocessing shell script to get that information. You only need to regenerate the information in the detections and gt_shasta folders. Make sure to change the arguments that say "cp" to your detection name. 

Additionally, to preprocess the LiDAR point clouds with 10 sweeps, you would run the following:
```bash
python tools/create_data.py nuscenes_data_prep --root_path="data/nuScenes" --save_path="data/nusc_preprocessed" --version="v1.0-trainval" --nsweeps=10
```