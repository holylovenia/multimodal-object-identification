def _resize_position_embeddings(model, new_num_tokens):
    vision_position_embedding = model.vision_model.embeddings.position_embedding
    text_position_embedding = model.text_model.embeddings.position_embedding
    new_vision_position_embedding = model._get_resized_embeddings(vision_position_embedding, new_num_tokens)
    new_text_position_embedding = model._get_resized_embeddings(text_position_embedding, new_num_tokens)
    model.vision_model.embeddings.position_embedding = new_vision_position_embedding
    model.text_model.embeddings.position_embedding = new_text_position_embedding
    model.config.update({"max_position_embeddings": new_num_tokens})
    return model