import torch
import clip

class ClipSimModel_Infer(torch.nn.Module):
    def __init__(self, clip_model_name, device, prompts=None):
        super(ClipSimModel_Infer, self).__init__()
        self.MMM, self.preprocess = clip.load(clip_model_name, device, jit=False)
        self.MMM.to(device)
        self.MMM.eval()

        labels_clip_prompt = ['positive', 'negative']
        # labels = ['unpleasant', 'pleasant']
        # labels = ['blameworthy', 'praiseworthy']
        text = clip.tokenize([f"This image is about something {labels_clip_prompt[0]}",
                              f"This image is about something {labels_clip_prompt[1]}"
                              ]).to(device)
        if prompts is not None:
            self.text_features = torch.HalfTensor(prompts).to(device)
            print('Using tuned prompts', self.text_features.shape)
        else:
            self.text_features = self.MMM.encode_text(text)
            
    def forward(self, x):
        image_features = self.MMM.encode_image(x)
        text_features_norm = self.text_features / self.text_features.norm(dim=-1, keepdim=True)
        # Pick the top 5 most similar labels for the image
        image_features_norm = image_features / image_features.norm(dim=-1, keepdim=True)
        similarity = (100.0 * image_features_norm @ text_features_norm.T)
        # values, indices = similarity[0].topk(5)
        return similarity.squeeze()