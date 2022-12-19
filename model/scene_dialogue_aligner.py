import torch
import torch.nn as nn
import transformers


class SceneDialogueAligner(nn.Module):
    def __init__(self, args, tokenizer):
        super(SceneDialogueAligner, self).__init__()
        self.args = args
        
        self.text_encoder = transformers.AutoModel.from_pretrained(self.args.text_model_name_or_path)
        # Fix model padding token id.
        self.text_encoder.resize_token_embeddings(len(tokenizer))

        self.vision_encoder = transformers.AutoModelForObjectDetection.from_pretrained(self.args.vision_model_name_or_path)
        
        self.text_fc = nn.Linear(self.text_encoder.config.hidden_size, self.args.hidden_size)
        self.visual_fc = nn.Linear(self.vision_encoder.config.d_model if self.vision_encoder.config.d_model else self.vision_encoder.config.hidden_size, self.args.hidden_size)

    def _get_scene_embeddings_from_vision_encoder(self, batch):
        pass

    def _get_dialogue_embeddings_from_text_encoder(self, batch):
        model_output = self.text_encoder(
            **batch["text_in"], output_hidden_states=True
        )
        last_hidden_state = model_output.last_hidden_state
        if "gpt2" in self.text_model_name_or_path:
            # Get the hidden state from the last token.
            # Code adopted from:
            # https://huggingface.co/transformers/v3.5.1/_modules/transformers/
            #           modeling_gpt2.html#GPT2ForSequenceClassification
            input_ids = batch["text_in"]["input_ids"]
            batch_size, sequence_length = input_ids.shape
            sequence_lengths = (
                torch.ne(input_ids, self.text_encoder.config.pad_token_id).sum(-1) - 1
            )
            text_embed = last_hidden_state[range(batch_size), sequence_lengths]
        elif "bert" in self.text_model_name_or_path:
            text_embed = last_hidden_state[:, 0, :]
        else:
            NotImplementedError("Only GPT2 and BERT has been implemented!")
        return text_embed

    def forward(self, batch):
        text_embed = self._get_dialogue_embeddings_from_text_encoder(batch)
        text_embed = self.text_fc(text_embed)

        visual_embed = self._get_scene_embeddings_from_vision_encoder(batch)

        # Compute cosine similarity.
        batch_logits = []
        for visual_feature, text_feature in zip(batch["features"], text_embed):
            visual_embed = self.visual_fc(visual_feature)
            logits = (visual_embed * text_feature).sum(axis=-1)
            batch_logits.append(logits)
        return batch_logits
