from typing import Any, Optional, Tuple, Union
from transformers.models.clip.modeling_clip import CLIPConfig, CLIPModel, CLIPOutput
from transformers.utils import (
    replace_return_docstrings,
)

import torch
import torch.nn as nn


# contrastive loss function, adapted from
# https://sachinruk.github.io/blog/pytorch/pytorch%20lightning/loss%20function/gpu/2021/03/07/CLIP.html
def contrastive_loss(
    logits: torch.Tensor, prefab_object_ids: torch.LongTensor,
    other_ambig_object_unique_ids: torch.LongTensor,
    include_other_similar_objects: bool,
    include_other_referred_objects: bool) -> torch.Tensor:
    
    # print("logits")
    # print(logits.shape, logits)

    labels = torch.eye(logits.shape[0]).to(logits.device)
    # 1 if it's the image of the same prefab object
    for i in range(labels.shape[0]):
        current_prefab_obj_id = prefab_object_ids[i]

        if include_other_similar_objects:
            # find indices of same elements in the tensor
            indices_of_same_prefab_objs = (
                prefab_object_ids == current_prefab_obj_id).nonzero(as_tuple=False)
            indices_of_same_prefab_objs = indices_of_same_prefab_objs.view(-1).tolist()
            # print("same", indices_of_same_prefab_objs)
        
        if include_other_referred_objects:
            indices_of_other_ambig_objs_in_dialogue = []
            for index, pid in enumerate(prefab_object_ids):
                current_other_ambig_prefab_obj_ids = torch.tensor(other_ambig_object_unique_ids[i])
                for other_id in current_other_ambig_prefab_obj_ids:
                    if pid == other_id:
                        indices_of_other_ambig_objs_in_dialogue.append(index)
            # print("other", indices_of_other_ambig_objs_in_dialogue)

        if include_other_similar_objects and include_other_referred_objects:
            indices_of_ambiguous_objs = list(set(indices_of_same_prefab_objs + indices_of_other_ambig_objs_in_dialogue))
        elif include_other_similar_objects:
            indices_of_ambiguous_objs = indices_of_same_prefab_objs
        elif include_other_referred_objects:
            indices_of_ambiguous_objs = indices_of_other_ambig_objs_in_dialogue
        
        for j in indices_of_ambiguous_objs:
            labels[i, j] = 1.
            labels[j, i] = 1.
    total_examples = torch.numel(labels)
    num_positive_examples = torch.count_nonzero(labels)
    pos_weight = (total_examples - num_positive_examples) / num_positive_examples
    return nn.functional.binary_cross_entropy_with_logits(logits, target=labels, pos_weight=pos_weight)

def clip_loss(
    similarity: torch.Tensor, prefab_object_ids: torch.LongTensor,
    other_ambig_object_unique_ids: torch.LongTensor,
    include_other_similar_objects: bool,
    include_other_referred_objects: bool) -> torch.Tensor:

    caption_loss = contrastive_loss(
        similarity, prefab_object_ids, other_ambig_object_unique_ids,
        include_other_similar_objects, include_other_referred_objects)
    image_loss = contrastive_loss(
        similarity.t(), prefab_object_ids, other_ambig_object_unique_ids,
        include_other_similar_objects, include_other_similar_objects)
    return (caption_loss + image_loss) / 2.0

class CLIPPERModel(CLIPModel):

    def modify_learning_objective(self, model_args):
        self.include_other_referred_objects = model_args.include_other_referred_objects
        self.include_other_similar_objects = model_args.include_other_similar_objects

    # @add_start_docstrings_to_model_forward(CLIP_INPUTS_DOCSTRING)
    @replace_return_docstrings(output_type=CLIPOutput, config_class=CLIPConfig)
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        object_ids: Optional[torch.LongTensor] = None,
        other_ambig_object_unique_ids: Optional[torch.LongTensor] = None,
        prefab_object_ids: Optional[torch.LongTensor] = None,
        return_loss: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CLIPOutput]:
        r"""
        Returns:
        Examples:
        ```python
        >>> from PIL import Image
        >>> import requests
        >>> from transformers import CLIPProcessor, CLIPModel
        >>> model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        >>> processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        >>> url = "http://images.cocodataset.org/val2017/000000039769.jpg"
        >>> image = Image.open(requests.get(url, stream=True).raw)
        >>> inputs = processor(
        ...     text=["a photo of a cat", "a photo of a dog"], images=image, return_tensors="pt", padding=True
        ... )
        >>> outputs = model(**inputs)
        >>> logits_per_image = outputs.logits_per_image  # this is the image-text similarity score
        >>> probs = logits_per_image.softmax(dim=1)  # we can take the softmax to get the label probabilities
        ```"""
        # Use CLIP model's config for some fields (if specified) instead of those of vision & text components.
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # print("object_ids", object_ids.shape, object_ids)
        # print()
        # print("prefab_object_ids", prefab_object_ids.shape, prefab_object_ids)
        # print()

        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        text_outputs = self.text_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        image_embeds = vision_outputs[1]
        image_embeds = self.visual_projection(image_embeds)

        text_embeds = text_outputs[1]
        text_embeds = self.text_projection(text_embeds)

        # normalized features
        image_embeds = image_embeds / image_embeds.norm(p=2, dim=-1, keepdim=True)
        text_embeds = text_embeds / text_embeds.norm(p=2, dim=-1, keepdim=True)

        # cosine similarity as logits
        logit_scale = self.logit_scale.exp()
        logits_per_text = torch.matmul(text_embeds, image_embeds.t()) * logit_scale
        logits_per_image = logits_per_text.t()

        loss = None
        if return_loss:
            loss = clip_loss(
                logits_per_text, prefab_object_ids, other_ambig_object_unique_ids,
                self.include_other_similar_objects, self.include_other_referred_objects)

        if not return_dict:
            output = (logits_per_image, logits_per_text, text_embeds, image_embeds, text_outputs, vision_outputs)
            return ((loss,) + output) if loss is not None else output

        return CLIPOutput(
            loss=loss,
            logits_per_image=logits_per_image,
            logits_per_text=logits_per_text,
            text_embeds=text_embeds,
            image_embeds=image_embeds,
            text_model_output=text_outputs,
            vision_model_output=vision_outputs,
        )