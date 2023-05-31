import os, argparse, numpy as np, json
from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--data_folder', type=str, default='data/nusc_preprocessed')
parser.add_argument('--detection_folder', type=str, default='data/nusc_preprocessed/test_2hz/detections')
parser.add_argument('--det_name', type=str, default='cp')
parser.add_argument('--file_name', type=str, default='test.json')
parser.add_argument('--velo', action='store_true', default=False)
parser.add_argument('--mode', type=str, default='2hz', choices=['20hz', '2hz'])
parser.add_argument('--split', type=str, default='test')
parser.add_argument('--test', action='store_true', default=False)
args = parser.parse_args()


def sample_result2bbox_array(sample):
    trans, size, rot, velocity, score = \
        sample['translation'], sample['size'],sample['rotation'], sample['velocity'], sample['detection_score']
    return trans + size + rot + velocity + [score]


def main(det_name, file_name, detection_folder, data_folder, mode):
    # dealing with the paths
    raw_file_path = os.path.join(data_folder, det_name, file_name)
    indiv_output_folder = os.path.join(detection_folder, det_name, 'individual_frames')
    if not os.path.exists(indiv_output_folder):
        os.makedirs(indiv_output_folder)


    cls_output_folder = os.path.join(detection_folder, det_name, 'cls_individual_frames')
    if not os.path.exists(cls_output_folder):
        os.makedirs(cls_output_folder)
    
    # load the detection file
    print('LOADING RAW FILE')
    f = open(raw_file_path, 'r')
    det_data = json.load(f)['results']
    f.close()

    # enumerate through all the frames
    sample_keys = list(det_data.keys())
    print('PROCESSING...')
    pbar = tqdm(total=len(sample_keys))
    for sample_key in sample_keys:
        # extract the bboxes and types
        sample_results = det_data[sample_key]
        with open(os.path.join(cls_output_folder, sample_key+'.json'), "w") as f:
            json.dump(sample_results, f)

        bboxes = []
        for sample in sample_results:
            bbox = sample_result2bbox_array(sample)
            bboxes.append(bbox)

        with open(os.path.join(indiv_output_folder, sample_key+'.json'), "w") as f:
            json.dump(bboxes, f)
        pbar.update(1)
    pbar.close()
    return


if __name__ == '__main__':
    data_folder = args.data_folder

    main(args.det_name, args.file_name, args.detection_folder, data_folder, args.mode)
