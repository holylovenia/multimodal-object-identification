import datasets
import json
import os

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
    for row in data:
        # Dialogue idx to labels
        dialog_id = row['dialog_id']
        turn_id = row['turn_id']
        labels = row['ambiguous_candidates']

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
            for obj in scene_objects['objects']:
                if obj["index"] in labels: # ambigous conv-image pair
                    other_ambig_object_ids = labels
                    other_ambig_object_ids.remove(obj['index'])

                    other_ambig_object_unique_ids = []
                    for id in other_ambig_object_ids:
                        for scene_obj in scene_objects["objects"]: # iterate over all objects in the scene
                            if scene_obj["index"] == id: # if they have the same index
                                other_ambig_object_unique_ids.append(scene_obj["unique_id"])
                    scene_dict[obj['index']] = (obj['bbox'], obj['unique_id'], other_ambig_object_unique_ids)

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
        if turn_id % 2 == 0:
            if utterance_turn == 'both' or utterance_turn =='user':
                utterances.append("<USER> " + turn)
        else:
            if utterance_turn == 'both' or utterance_turn =='system':
                utterances.append("<SYS> " + turn)
    example_batch['caption'] = (" ".join(utterances[-num_utterances:])).lower()
    return example_batch

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

def add_empty_dialogue(example_batch):
    if 'dialogue' not in example_batch:
        example_batch['dialogue'] = ['' for i in range(10)]
    return example_batch

def tokenize_text(example_batch, tokenizer, text_column_name='caption'):
    example_batch['input_ids'] = tokenizer(example_batch[text_column_name])['input_ids']
    return example_batch

###
#
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
        'dialog_id': [], 'scene_id': [], 'turn_id': [], 
        'dialogue': [], 'image': [], 'objects': []
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
                scene_dict[index_mapping[obj['index']]] = obj['bbox']

        image_path = row['image_name']
        for img_dir_path in img_dir_paths:
            if os.path.exists(f'{img_dir_path}/{image_path}'):
                image_path = f'{img_dir_path}/{image_path}'
                break

        dialogue = row["input_text"]
        for obj_id, bbox in scene_dict.items():
            dset['dialog_id'].append(dialog_id)
            dset['scene_id'].append(scene_id)
            dset['turn_id'].append(turn_id)
            dset['object_id'].append(obj_id)
            dset['labels'].append(labels)
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
