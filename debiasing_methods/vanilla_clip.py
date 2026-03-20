from .method_utils import *


class VanillaCLIP(CLIPEmbeddingDebiasingMethod):
    def debias_text(self, text_prompts, *args, **kwargs):
        with torch.no_grad():
            query_embedding = get_clip_text_embedding(text_prompts, self.model, normalize=False).float()
        return query_embedding

    def debias_images(self, image_embeddings, *args, **kwargs):
        return image_embeddings.float()
