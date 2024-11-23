import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel, BertModel
from scipy.stats import special_ortho_group

def get_elmes(embedding_dim, num_classes, device):
    """
    
    This function creates a set of vectors that are equal in length and maximally
    equiangular, which can be used for encoding class labels in a way that allows
    for better class separation in the embedding space.
    
    Args:
    embedding_dim (int): Dimension of the embedding vectors.
    num_classes (int): Number of classes to generate embeddings for.
    device (torch.device): Device to place the resulting tensor on.
    
    Returns:
    torch.Tensor: ELMES embeddings of shape (num_classes, embedding_dim)
    """
    
    np.random.seed(42)
    
    #generate a random orthogonal matrix
    #so vectors start off orthogonal to each other
    ortho_matrix = special_ortho_group.rvs(embedding_dim)[:num_classes]
    
    #calculate the angle between vectors for maximal equiangularity
    optimal_angle = np.arccos(-1 / (num_classes - 1))
    
    #aAdjust the orthogonal vectors to be maximally equiangular
    adjustment_factor = np.sqrt((1 - np.cos(optimal_angle)) / 2)
    adjusted_matrix = ortho_matrix * adjustment_factor
    
    #add a constant to all elements to shift the vectors
    #so they're not centered around the origin
    constant_shift = np.sqrt(1 - adjustment_factor**2) / np.sqrt(num_classes)
    elmes_matrix = adjusted_matrix + constant_shift
    
    #normalize the vectors to ensure equal length
    elmes_matrix /= np.linalg.norm(elmes_matrix, axis=1, keepdims=True)
    
    elmes_embeddings = torch.tensor(elmes_matrix, dtype=torch.float32).to(device)
    
    return elmes_embeddings

class CAML(nn.Module):
    def __init__(self, num_classes, hidden_dim=768, device=torch.device('cuda:0')):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        

        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
    
        self.label_elmes = nn.Parameter(get_elmes(hidden_dim, num_classes, device), requires_grad=False)
        self.elmes_scale = nn.Parameter(torch.ones(1))
        
        

        self.transformer_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, images, labels):
        image_features = self.image_encoder(images).last_hidden_state
        
   
        label_one_hot = nn.functional.one_hot(labels, num_classes=self.num_classes).float()
        label_embeddings = label_one_hot @ (self.elmes_scale * self.label_elmes)
        
        
        
        combined_input = torch.cat([image_features, label_embeddings.unsqueeze(1)], dim=1)
        
        

        encoder_output = self.transformer_encoder(inputs_embeds=combined_input).last_hidden_state
        
        
        logits = self.classifier(encoder_output[:, 0, :])  # Use [CLS] token for classification
        
        return logits

    def meta_test(self, support_images, support_labels, query_images, way, shot):
        #[way, shot, 3, 224, 224] -> [way*shot, 3, 224, 224]
        support_images = support_images.view(-1, *support_images.shape[2:])
        
        #[1, way, 3, 224, 224] -> [way, 3, 224, 224]
        query_images = query_images.squeeze(0)


        support_features = self.image_encoder(support_images).last_hidden_state[:, 0]
        query_features = self.image_encoder(query_images).last_hidden_state[:, 0]


        support_features = support_features.view(way, shot, -1)
        support_labels = support_labels.view(way, shot)

        
        prototypes = support_features.mean(dim=1)  # [way, feature_dim]

        
        dists = torch.cdist(query_features, prototypes)

        
        return -dists

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)

