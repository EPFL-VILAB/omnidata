import pickle
from tqdm import tqdm
from torch.utils.data import DataLoader
from data.taskonomy_replica_gso_dataset import TaskonomyReplicaGsoDataset

def save_pickle(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
        
def load_pickle(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

    
# Create datasets
dataset_names = ['habitat2'] # 'taskonomy', 'hypersim', 'replica', 'gso', 'habitat2'
splits = ['train', 'val'] # 'train', 'val', 'test'
image_size = 512
batch_size = 16

datasets = []
for dataset_name in dataset_names:
    for split in splits:
        print(f'Preparing {dataset_name} / {split}')
        options = TaskonomyReplicaGsoDataset.Options(
            taskonomy_variant='fullplus',
            split=split,
            tasks=['point_info'],
            datasets=[dataset_name],
            transform='DEFAULT',
            image_size=image_size,
            normalize_rgb=False,
            randomize_views=False
        )
        dataset = TaskonomyReplicaGsoDataset(options)
        loader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            num_workers=batch_size, 
            collate_fn=lambda x: x
        )
        dataset_dict = {
            'dataset': dataset,
            'loader': loader,
            'name': dataset_name,
            'split': split
        }
        datasets.append(dataset_dict)
        
        
for dataset_dict in datasets:
    print(f'Gathering stats for {dataset_dict["name"]} {dataset_dict["split"]}')
    
    rots_x = []
    rots_y = []
    rots_z = []
    fovs = []
    heights = []
    widths = []
    if dataset_dict['name'] != 'hypersim':
        distances = []
        obliqueness_angles = []
        points_in_views = []
        views_per_point = {}
        points_per_camera = {}
        cameras_per_point = {}

    pbar = tqdm(total=len(dataset_dict['loader']))
    for batch in dataset_dict['loader']:
        for sample in batch:
            point_info = sample['positive']['point_info']
            point = sample['positive']['point']
            view = sample['positive']['view']
            building = sample['positive']['building']

            if dataset_dict['name'] != 'hypersim':
                rot_x, rot_y, rot_z = point_info['camera_rotation_final']
                camera_fov = point_info['field_of_view_rads']
                height = point_info['resolution']
                width = point_info['resolution']
                camera_distance = point_info['camera_distance']
                obliqueness_angle = point_info['obliqueness_angle']
                points_in_view = len(point_info['nonfixated_points_in_view'])
            else:
                rot_x, rot_y, rot_z = point_info['rotation']
                camera_fov = point_info['fov_x']
                height = point_info['height']
                width = point_info['width']
            
            rots_x.append(rot_x)
            rots_y.append(rot_y)
            rots_z.append(rot_z)
            fovs.append(camera_fov)
            heights.append(height)
            widths.append(width)
            if dataset_dict['name'] != 'hypersim':
                distances.append(camera_distance)
                obliqueness_angles.append(obliqueness_angle)
                points_in_views.append(points_in_view)

                # Views per point
                if (building, point) not in views_per_point:
                    views_per_point[(building, point)] = [view]
                else:
                    views_per_point[(building, point)].append(view)

                # Points per camera (i.e. how many other points does one camera see across all views)
                if (building, point) not in points_per_camera:
                    points_per_camera[(building, point)] = set(point_info['nonfixated_points_in_view'])
                else:
                    points_per_camera[(building, point)] = points_per_camera[(building, point)].union(
                        point_info['nonfixated_points_in_view']
                    )

                # Cameras per point (i.e. by how many other cameras is one point seen?)
                for other_point in point_info['nonfixated_points_in_view']:
                    if (building, other_point) not in cameras_per_point:
                        cameras_per_point[(building, other_point)] = set(point)
                    else:
                        cameras_per_point[(building, other_point)] = cameras_per_point[(building, other_point)].union(point)

        pbar.update(1)
    pbar.close()
    
    if dataset_dict['name'] != 'hypersim':
        views_per_point = [len(v) for v in views_per_point.values()]
        points_per_camera = [len(v) for v in points_per_camera.values()]
        cameras_per_point = [len(v) for v in cameras_per_point.values()]
    
    save_pickle(rots_x, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-rots_x.pkl')
    save_pickle(rots_y, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-rots_y.pkl')
    save_pickle(rots_z, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-rots_z.pkl')
    save_pickle(fovs, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-fovs.pkl')
    if dataset_dict['name'] != 'hypersim':
        save_pickle(distances, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-distances.pkl')
        save_pickle(obliqueness_angles, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-obliqueness_angles.pkl')
        save_pickle(points_in_views, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-points_in_views.pkl')
        save_pickle(views_per_point, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-views_per_point.pkl')
        save_pickle(points_per_camera, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-points_per_camera.pkl')
        save_pickle(cameras_per_point, f'./omnidata_stats/{dataset_dict["name"]}-{dataset_dict["split"]}-cameras_per_point.pkl')