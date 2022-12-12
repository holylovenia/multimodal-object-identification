import datasets
import json
import numpy as np
import os
import PIL
import torch

###
# Load Object Categories in SIMMC Dataset
###
def load_categories(
    fashion_prefab_path="./simmc2/data/fashion_prefab_metadata_all.json",
    furniture_prefab_path="./simmc2/data/furniture_prefab_metadata_all.json",
    return_simple_mapping=True):
    
    fashion_prefab = json.loads(open(fashion_prefab_path).read())
    furniture_prefab = json.loads(open(furniture_prefab_path).read())
    
    categories = []
    fashion_categories = list(set([fashion_prefab[item]["type"] for item in fashion_prefab]))
    furniture_categories = list(set([furniture_prefab[item]["type"] for item in furniture_prefab]))
    category_mapping = {}
    
    for i in range(len(fashion_categories)):
        categories.append({
            "supercategory": "fashion",
            "id": i,
            "name": fashion_categories[i],
        })
                
    current_num_categories = len(categories)
    for i in range(len(furniture_categories)):
        categories.append({
            "supercategory": "furniture",
            "id": i + current_num_categories,
            "name": furniture_categories[i],
        })
        
    if return_simple_mapping:
        id2cat = dict(enumerate(fashion_categories + furniture_categories))
        cat2id = {v: k for k, v in id2cat.items()}
        return {"categories": categories, "id2cat": id2cat, "cat2id": cat2id}
    
    return categories

###
# Load Objects in Scene for DETR Training
###
def load_objects_in_scenes_dataset(
    mapping,
    img_dir_paths=[
        "./simmc2/data/simmc2_scene_images_dstc10_public_part1",
        "./simmc2/data/simmc2_scene_images_dstc10_public_part2"],
    scene_dir_path="./simmc2/data/public",
    fashion_prefab_path="./simmc2/data/fashion_prefab_metadata_all.json",
#     include_fashion_attrs=["type", "color", "pattern", "sleeveLength"],
    furniture_prefab_path="./simmc2/data/furniture_prefab_metadata_all.json",
#     include_furniture_attrs=["type", "color", "materials"],
    ):
    
    data_dict = {
        "image": [], "image_id": [], "objects": []}
    
    fashion_prefab = json.loads(open(fashion_prefab_path).read())
    furniture_prefab = json.loads(open(furniture_prefab_path).read())
    scene2id = {}
    
    for img_dir_path in img_dir_paths:
        
        for img_file_id in os.listdir(img_dir_path):
            img_file_path = os.path.join(img_dir_path, img_file_id)
            scene_id = img_file_id.split(".")[0]

            if os.path.exists(f"{scene_dir_path}/{scene_id}_scene.json"):
                scene_file_path = f"{scene_dir_path}/{scene_id}_scene.json"
            elif os.path.exists(f"{scene_dir_path}/m_{scene_id}_scene.json"):
                scene_file_path = f"{scene_dir_path}/m_{scene_id}_scene.json"
                
            if os.path.isfile(scene_file_path):
                data_dict["image"].append(img_file_path)
                num_scenes = len(data_dict["image_id"])
                scene2id[scene_id] = num_scenes
                data_dict["image_id"].append(num_scenes)
                
                scene_json = json.loads(open(scene_file_path).read())
                scene_objects = scene_json["scenes"][0]["objects"]
                objects = []
                for scene_object in scene_objects:
                    object_annotation = {
                        "bbox": [float(b) for b in scene_object["bbox"]],
                        "id": scene_object["unique_id"],
                        "area": None,
                        "segmentation": [],
                        "iscrowd": False,
                    }
                    if fashion_prefab.get(scene_object["prefab_path"]) is not None:
                        item = fashion_prefab[scene_object["prefab_path"]]
                    else:
                        item = furniture_prefab[scene_object["prefab_path"]]
                    object_annotation["category_id"] = mapping["cat2id"][item["type"]]
                    objects.append(object_annotation)
                data_dict["objects"].append(objects)
            
    dataset = datasets.Dataset.from_dict(data_dict)
    dataset = dataset.cast_column("image", datasets.Image(decode=True))
    
    id2scene = {v: k for k, v in scene2id.items()}
    mapping["scene2id"] = scene2id
    mapping["id2scene"] = id2scene
    
    return dataset, mapping


def compute_image_area(example_batch):
    width, height = example_batch["image"].size
    area = width * height
    for i in range(len(example_batch["objects"])):
        example_batch["objects"][i]["area"] = area
    return example_batch

###
# Load Image Text Dataset for CLIP Fine-Tuning
###
def load_image_text_dataset(
    img_dir_paths=[
        "./simmc2/data/simmc2_scene_images_dstc10_public_part1",
        "./simmc2/data/simmc2_scene_images_dstc10_public_part2"],
    scene_dir_path="./simmc2/data/public",
    fashion_prefab_path="./simmc2/data/fashion_prefab_metadata_all.json",
    include_fashion_attrs=["type", "color", "pattern", "sleeveLength"],
    furniture_prefab_path="./simmc2/data/furniture_prefab_metadata_all.json",
    include_furniture_attrs=["type", "color", "materials"]):

    data_dict = {"object_id": [], "image": [], "bbox": [], "attrs": []}

    fashion_prefab = json.loads(open(fashion_prefab_path).read())
    furniture_prefab = json.loads(open(furniture_prefab_path).read())
    
    for img_dir_path in img_dir_paths:
        for img_file_id in os.listdir(img_dir_path):
            img_file_path = os.path.join(img_dir_path, img_file_id)
            scene_id = img_file_id.split(".")[0]                

            if os.path.exists(f"{scene_dir_path}/{scene_id}_scene.json"):
                scene_file_path = f"{scene_dir_path}/{scene_id}_scene.json"
            elif os.path.exists(f"{scene_dir_path}/m_{scene_id}_scene.json"):
                scene_file_path = f"{scene_dir_path}/m_{scene_id}_scene.json"

            scene_json = json.loads(open(scene_file_path).read())
            scene_objects = scene_json["scenes"][0]["objects"]
            objects = []
            for scene_object in scene_objects:
                # In the case of invalid width or height
                if scene_object["bbox"][2] == 0 or scene_object["bbox"][3] == 0:
                    continue

                data_dict["image"].append(img_file_path)
                data_dict["bbox"].append(scene_object["bbox"])
                data_dict["object_id"].append(scene_object["unique_id"])

                attrs = {}
                if fashion_prefab.get(scene_object["prefab_path"]) is not None:
                    item = fashion_prefab[scene_object["prefab_path"]]
                    for f_attr in include_fashion_attrs:
                        attrs[f_attr] = item[f_attr]
                else:
                    item = furniture_prefab[scene_object["prefab_path"]]
                    for f_attr in include_furniture_attrs:
                        attrs[f_attr] = item[f_attr]
                data_dict["attrs"].append(attrs)

    dataset = datasets.Dataset.from_dict(data_dict)
    dataset = dataset.cast_column("image", datasets.Image(decode=True))
    return dataset


def convert_attrs_to_caption(example_batch):
    caption = example_batch["attrs"]["color"]
    if example_batch["attrs"]["pattern"] is not None:
        caption = caption + " " + example_batch["attrs"]["pattern"] + "-patterned"
    if example_batch["attrs"]["materials"] is not None:
        caption = caption + " " + example_batch["attrs"]["materials"]
    if example_batch["attrs"]["sleeveLength"] is not None:
        caption = caption + " " + example_batch["attrs"]["sleeveLength"] + "-sleeved"
    caption = caption + " " + example_batch["attrs"]["type"]
    example_batch["caption"] = caption.lower()
    return example_batch


def tokenize_captions(example_batch, tokenizer, max_seq_length):
    text_inputs = tokenizer(
        example_batch["caption"], max_length=max_seq_length, padding="max_length", truncation=True)
    example_batch["input_ids"] = text_inputs.input_ids
    example_batch["attention_mask"] = text_inputs.attention_mask
    return example_batch

###
# Dataset for CLIP training using conversation data
###
def load_image_conv_dataset(
    scene_dir_path = "./simmc2/data/public", 
    data_path = './preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json',
    img_dir_paths = [
        './simmc2/data/simmc2_scene_images_dstc10_public_part1',
        './simmc2/data/simmc2_scene_images_dstc10_public_part2'
    ],
    return_gt_labels=True,
):
    with open(data_path, "r") as file_id:
        raw_data = json.load(file_id)    
    data = raw_data["data"]
    gold_data = json.load(open(raw_data['source_path'],'r'))
    
    dset = {
        'dialog_id': [], 'scene_id': [], 'turn_id': [], 'object_id': [],
        'prefab_object_id': [], 'other_ambig_object_unique_ids': [],
        'dialogue': [], 'image': [], 'bbox': []
    }
    
    for row_id, row in enumerate(data):
        # Dialogue idx to labels
        dialog_id = row['dialog_id']
        turn_id = row['turn_id']
        labels = row['ambiguous_candidates']
        object_map = row['object_map']

        # Scene
        scene_path = row['image_name'].replace('.png','_scene.json')
        scene_id = row['image_name'].split(".")[0]
        if os.path.exists(f"{scene_dir_path}/{scene_path}"):
            scene_path = f"{scene_dir_path}/{scene_path}"
        elif os.path.exists(f"{scene_dir_path}/m_{scene_path}"):
            scene_path = f"{scene_dir_path}/m_{scene_path}"

        scene = json.load(open(scene_path, 'r'))
        scene_dict = {}

        for scene_objects in scene['scenes']:
            
            index_mapping = {obj['index']: local_object_id \
                for local_object_id, obj in zip(object_map, scene_objects['objects'])}
            local_to_prefab_index_mapping = {local_object_id: obj['unique_id'] \
                for local_object_id, obj in zip(object_map, scene_objects['objects'])}

            for obj in scene_objects['objects']:
                local_object_id = index_mapping[obj['index']]
                if local_object_id in labels: # ambigous conv-image pair
                    other_ambig_object_ids = labels.copy()
                    other_ambig_object_ids.remove(local_object_id)

                    other_ambig_object_unique_ids = []
                    for local_id in other_ambig_object_ids:
                        if local_id in local_to_prefab_index_mapping:
                            other_ambig_object_unique_ids.append(local_to_prefab_index_mapping[local_id])
                        else:
                            other_ambig_object_unique_ids.append(local_id)
                            
                    scene_dict[index_mapping[obj['index']]] = (obj['bbox'], obj['unique_id'], other_ambig_object_unique_ids)
                
                # if dialog_id == 11496:
                #     print(dialog_id, turn_id, obj_id, other_ambig_object_unique_ids, dialogue)

        image_path = row['image_name']
        for img_dir_path in img_dir_paths:
            if os.path.exists(f'{img_dir_path}/{image_path}'):
                image_path = f'{img_dir_path}/{image_path}'
                break

        dialogue = row["input_text"]
        for obj_id, (bbox, prefab_obj_id, other_ambig_object_unique_ids) in scene_dict.items():
            dset['dialog_id'].append(dialog_id)
            dset['scene_id'].append(scene_id)
            dset['turn_id'].append(turn_id)
            dset['object_id'].append(obj_id)
            dset['prefab_object_id'].append(prefab_obj_id)
            dset['other_ambig_object_unique_ids'].append(other_ambig_object_unique_ids)
            dset['dialogue'].append(dialogue)
            dset['image'].append(image_path)
            dset['bbox'].append(bbox)
            
    eval_dset = datasets.Dataset.from_dict(dset)
    eval_dset = eval_dset.cast_column("image", datasets.Image(decode=True))
    
    if return_gt_labels:
        return eval_dset, gold_data
    else:
        return eval_dset

###
# Load Dataset for CLIP Evaluation
###
def load_image_text_eval_dataset(
    scene_dir_path = "./simmc2/data/public", 
    data_path = './preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json',
    img_dir_paths = [
        './simmc2/data/simmc2_scene_images_dstc10_public_part1',
        './simmc2/data/simmc2_scene_images_dstc10_public_part2'
    ],
    return_gt_labels=True,
):
    with open(data_path, "r") as file_id:
        raw_data = json.load(file_id)    
    data = raw_data["data"]
    gold_data = json.load(open(raw_data['source_path'],'r'))
    
    dset = {
        'dialogue': [], 'image': [], 'bbox': []
    }
    meta_dset = {
        'dialog_id': [], 'scene_id': [], 'turn_id': [], 'object_id': [], 'labels': []
    }
    for row in data:
        # Dialogue idx to labels
        dialog_id = row['dialog_id']
        turn_id = row['turn_id']
        labels = row['ambiguous_candidates']
        object_map = row['object_map']

        # Scene
        scene_path = row['image_name'].replace('.png','_scene.json')
        scene_id = row['image_name'].split(".")[0]
        if os.path.exists(f"{scene_dir_path}/{scene_path}"):
            scene_path = f"{scene_dir_path}/{scene_path}"
        elif os.path.exists(f"{scene_dir_path}/m_{scene_path}"):
            scene_path = f"{scene_dir_path}/m_{scene_path}"

        scene = json.load(open(scene_path, 'r'))
        scene_dict = {}
        for scene_objects in scene['scenes']:
            index_mapping = {}
            for obj_id, obj in zip(object_map, scene_objects['objects']):
                index_mapping[obj['index']] = obj_id
            
            for obj in scene_objects['objects']:
                # if index_mapping[obj['index']] in labels:
                scene_dict[index_mapping[obj['index']]] = obj['bbox']

        image_path = row['image_name']
        for img_dir_path in img_dir_paths:
            if os.path.exists(f'{img_dir_path}/{image_path}'):
                image_path = f'{img_dir_path}/{image_path}'
                break

        dialogue = row["input_text"]
        for obj_id, bbox in scene_dict.items():
            meta_dset['dialog_id'].append(dialog_id)
            meta_dset['scene_id'].append(scene_id)
            meta_dset['turn_id'].append(turn_id)
            meta_dset['object_id'].append(obj_id)
            meta_dset['labels'].append(labels)
            dset['dialogue'].append(dialogue)
            dset['image'].append(image_path)
            dset['bbox'].append(bbox)
            
    meta_dset = datasets.Dataset.from_dict(meta_dset)
    eval_dset = datasets.Dataset.from_dict(dset)
    eval_dset = eval_dset.cast_column("image", datasets.Image(decode=True))
    
    if return_gt_labels:
        return eval_dset, meta_dset, gold_data
    else:
        return eval_dset

def convert_dialogue_to_caption(example_batch, num_utterances=3, utterance_turn='both'):
    utterances = []
    for turn_id, turn in enumerate(example_batch['dialogue']):
        if len(turn) == 0:
            continue # Skip empty string

        if turn_id % 2 == 0:
            if utterance_turn == 'both' or utterance_turn =='user':
                utterances.append("<USER> " + turn)
        else:
            if utterance_turn == 'both' or utterance_turn =='system':
                utterances.append("<SYS> " + turn)
    example_batch['caption'] = (" ".join(utterances[-num_utterances:])).lower()
    return example_batch

def add_sitcom_detr_attr(example_batch):
    for object_ in example_batch['objects']:
        if 'index' not in object_:
            object_['index'] = 0
        
    if 'dialog_id' not in example_batch:
        example_batch['dialog_id'] = 0
    if 'turn_id' not in example_batch:
        example_batch['turn_id'] = 0
    if 'dialogue' not in example_batch:
        example_batch['dialogue'] = ['' for i in range(10)]
    if 'all_objects' not in example_batch:
        example_batch['all_objects'] = example_batch['objects']
        
    return example_batch

def add_turn_dialog_id(example_batch):
    return example_batch

def tokenize_text(example_batch, tokenizer, text_column_name='caption'):
    example_batch['input_ids'] = tokenizer(example_batch[text_column_name])['input_ids']
    return example_batch

###
#
###
def load_sitcom_detr_dataset(
    mapping,
    scene_dir_path = "./simmc2/data/public", 
    data_path = './preprocessed_data/ambiguous_candidates/simmc2.1_ambiguous_candidates_dstc11_devtest.json',
    img_dir_paths = [
        './simmc2/data/simmc2_scene_images_dstc10_public_part1',
        './simmc2/data/simmc2_scene_images_dstc10_public_part2'
    ],
    fashion_prefab_path="./simmc2/data/fashion_prefab_metadata_all.json",
    furniture_prefab_path="./simmc2/data/furniture_prefab_metadata_all.json",    
    return_gt_labels=True,
):
    with open(data_path, "r") as file_id:
        raw_data = json.load(file_id)    
    data = raw_data["data"]
    gold_data = json.load(open(raw_data['source_path'],'r'))

    fashion_prefab = json.loads(open(fashion_prefab_path).read())
    furniture_prefab = json.loads(open(furniture_prefab_path).read())

    dset = {
        # 'dialog_id': [], 'scene_id': [], 'turn_id': [], 
        'image': [], 'image_id': [], 'objects': [], 'dialogue': [],
        'turn_id': [], 'dialog_id': [], 'all_objects': []
    }

    for i, row in enumerate(data):
        # Dialogue idx to labels
        dialog_id = int(row['dialog_id'])
        turn_id = int(row['turn_id'])
        labels = row['ambiguous_candidates']
        object_map = row['object_map']
        # Scene
        scene_path = row['image_name'].replace('.png','_scene.json')
        scene_id = row['image_name'].split(".")[0]
        if os.path.exists(f"{scene_dir_path}/{scene_path}"):
            scene_path = f"{scene_dir_path}/{scene_path}"
        elif os.path.exists(f"{scene_dir_path}/m_{scene_path}"):
            scene_path = f"{scene_dir_path}/m_{scene_path}"

        scene = json.load(open(scene_path, 'r'))
        scene_dict = {}
        
        assert len(scene['scenes']) == 1 # Ensure there is only 1 scene
        scene_objects = scene['scenes'][0] 
        
        index_mapping = {}
        for obj_id, obj in zip(object_map, scene_objects['objects']):
            index_mapping[obj['index']] = obj_id

        all_objects = []
        objects = []
        for scene_object in scene_objects['objects']:
            object_annotation = {
                "bbox": [float(b) for b in scene_object["bbox"]],
                "id": scene_object["unique_id"],
                "index": index_mapping[scene_object['index']],
                "area": None,
                "segmentation": [],
                "iscrowd": False,
            }
            if fashion_prefab.get(scene_object["prefab_path"]) is not None:
                item = fashion_prefab[scene_object["prefab_path"]]
            else:
                item = furniture_prefab[scene_object["prefab_path"]]
            object_annotation["category_id"] = mapping["cat2id"][item["type"]]
            if index_mapping[scene_object['index']] in labels:
                objects.append(object_annotation)
            all_objects.append(object_annotation)

        dialogue = row["input_text"]
        image_path = row['image_name']
        for img_dir_path in img_dir_paths:
            if os.path.exists(f'{img_dir_path}/{image_path}'):
                image_path = f'{img_dir_path}/{image_path}'
                break

        # dset['scene_id'].append(scene_id)
        dset['dialog_id'].append(dialog_id)
        dset['turn_id'].append(turn_id)
        dset['image'].append(image_path)
        dset['image_id'].append(i)
        dset['objects'].append(objects)
        dset['dialogue'].append(dialogue)        
        dset['all_objects'].append(all_objects)

    dset = datasets.Dataset.from_dict(dset)
    dset = dset.cast_column("image", datasets.Image(decode=True))

    if return_gt_labels:
        return dset, gold_data
    else:
        return dset


class Dataloader:
    def __init__(
        self, tokenizer, load_path, args, hidden_labels=False,
        img_dir_paths=[
            "./simmc2/data/simmc2_scene_images_dstc10_public_part1",
            "./simmc2/data/simmc2_scene_images_dstc10_public_part2"],
        scene_dir_path="./simmc2/data/public"):
        self._tokenizer = tokenizer
        # self._features = feature_loader
        self._args = args
        self._hidden_labels = hidden_labels
        self._img_dir_paths = img_dir_paths
        self._scene_dir_path = scene_dir_path
        print("Loading: {}".format(load_path))
        with open(load_path, "r") as file_id:
            self._raw_data = json.load(file_id)
        # Also read the source data for evaluation.
        with open(self._raw_data["source_path"], "r") as file_id:
            self.source_data = json.load(file_id)
        self._data = self._raw_data["data"]

        self.num_utterances = 2 * args.max_turns + 1
        self.num_instances = len(self._data)

    def get_random_batch(self, batch_size):
        indices = np.random.randint(0, self.num_instances, batch_size)
        return self.get_indexed_data(indices)

    def get_entire_batch(self, batch_size):
        all_indices = np.arange(self.num_instances)
        for start in all_indices[::batch_size]:
            batch_indices = all_indices[start : start + batch_size]
            yield self.get_indexed_data(batch_indices)

    def get_indexed_data(self, indices):
        text_labels = []
        text_inputs = []
        dialog_ids = []
        turn_ids = []
        # features = []
        object_maps = []
        object_annotations = []
        scene_image_paths = []
        scene_images = []
        for index in indices:
            # Add <USER> and <SYS> tokens.
            dialog_datum = self._data[index]
            dialog = self._data[index]["input_text"]
            for turn_id, turn in enumerate(dialog):
                if turn_id % 2 == 0:
                    dialog[turn_id] = "<USER> " + turn
                else:
                    dialog[turn_id] = "<SYS> " + turn
            text = " ".join(dialog[-self.num_utterances :])
            text_inputs.append(text)
            dialog_ids.append(dialog_datum["dialog_id"])
            turn_ids.append(dialog_datum["turn_id"])
            object_map = dialog_datum["object_map"]
            scene_image_name = dialog_datum["image_name"]
            # # Get image features.
            # if self._features:
            #     features.append(self._features[dialog_datum["image_name"]])
            #     num_candidates = features[-1].shape[0]
            # else:
            num_candidates = len(object_map)

            # Get the ambiguous candidates and map it local ids.
            global_ambiguous_candidates = dialog_datum["ambiguous_candidates"]
            for candidate in global_ambiguous_candidates:
                if candidate not in object_map:
                    object_map.append(candidate)
            local_ambiguous_candidates = [
                object_map.index(ii) for ii in global_ambiguous_candidates
            ]
            # NOTE: Few scenes have misaligned image features (ignore them).
            # Default object map to linear local ids in such cases.
            if len(object_map) != num_candidates:
                local_ambiguous_candidates = global_ambiguous_candidates
                object_map = list(range(num_candidates))

            img_file_path = os.path.join(self._img_dir_paths[0], scene_image_name)
            if not os.path.exists(img_file_path):
                img_file_path = os.path.join(self._img_dir_paths[1], scene_image_name)
            scene_id = scene_image_name.split(".")[0]

            if os.path.exists(f"{self._scene_dir_path}/{scene_id}_scene.json"):
                scene_file_path = f"{self._scene_dir_path}/{scene_id}_scene.json"
            elif os.path.exists(f"{self._scene_dir_path}/m_{scene_id}_scene.json"):
                scene_file_path = f"{self._scene_dir_path}/m_{scene_id}_scene.json"
                
            if os.path.isfile(scene_file_path):
                scene_image_paths.append(img_file_path)
                scene_image = PIL.Image.open(img_file_path)
                scene_images.append(scene_image)

                width, height = scene_image.size
                area = width * height
                
                scene_json = json.loads(open(scene_file_path).read())
                scene_objects = scene_json["scenes"][0]["objects"]

                object_annotation = [{
                    "bbox": [float(b) for b in scene_object["bbox"]],
                    "id": scene_object["unique_id"],
                    "area": area,
                    "segmentation": [],
                    "iscrowd": False,
                } for scene_object in scene_objects]

            object_maps.append(object_map)
            object_annotations.append(object_annotation)
            label = torch.tensor(
                [
                    1 if ii in local_ambiguous_candidates else 0
                    for ii in range(num_candidates)
                ],
                dtype=torch.float32,
            )
            text_labels.append(label)

        encoded_inputs = self._tokenizer(
            text_inputs,
            padding=True,
            max_length=self._args.max_seq_length,
            return_tensors="pt",
            truncation=True,
        )
        encoded_inputs = {key: val.detach() for key, val in encoded_inputs.items()}
        if self._hidden_labels:
            # Reset all the text_labels to [0] (dummy labels).
            text_labels = [[0] for ii in text_labels]

        # Pack the batch.
        batch = {
            "text_in": encoded_inputs,
            "gt_label": text_labels,
            "dialog_id": dialog_ids,
            "turn_id": turn_ids,
            # "features": features,
            "object_map": object_maps,
            "object_annotation": object_annotations,
            "scene_image_path": scene_image_paths,
            "scene_image": scene_images,
        }
        return batch