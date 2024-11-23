import torch
import torch.nn as nn
import numpy as np
from transformers import ViTModel, BertModel
from scipy.stats import special_ortho_group
import torch.nn.functional as F

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

def get_hd_vectors(embedding_dim, num_classes, device):
    """Generate hyperdimensional vectors for class encoding."""
    hd_vectors = torch.randint(0, 2, (num_classes, embedding_dim), dtype=torch.float32, device=device)
    hd_vectors = 2 * hd_vectors - 1  # Convert to {-1, 1}
    return hd_vectors

class HybridEncoder(nn.Module):
    def __init__(self, num_classes, elmes_dim, hdc_dim, device):
        super().__init__()
        self.elmes_vectors = nn.Parameter(get_elmes(elmes_dim, num_classes, device), requires_grad=False)
        self.hdc_vectors = nn.Parameter(get_hd_vectors(hdc_dim, num_classes, device), requires_grad=False)
        self.elmes_scale = nn.Parameter(torch.ones(1))
        self.hdc_scale = nn.Parameter(torch.ones(1))
        self.combine = nn.Linear(elmes_dim + hdc_dim, elmes_dim)

    def forward(self, labels):
        elmes_repr = self.elmes_scale * self.elmes_vectors[labels]
        hdc_repr = self.hdc_scale * self.hdc_vectors[labels]
        combined = torch.cat([elmes_repr, hdc_repr], dim=-1)
        return self.combine(combined)

    def regularization_loss(self):
        l2_loss = torch.sum(self.combine.weight ** 2)
        

        hdc_usage = torch.sum(self.hdc_vectors, dim=0)
        entropy_loss = -torch.sum(F.softmax(hdc_usage, dim=0) * F.log_softmax(hdc_usage, dim=0))
        
        return l2_loss + entropy_loss


class CAML(nn.Module):
    def __init__(self, num_classes, hidden_dim=768, hdc_dim=256, device=torch.device('cuda:0')):
        super().__init__()
        self.device = device
        self.num_classes = num_classes
        

        self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224')
        for param in self.image_encoder.parameters():
            param.requires_grad = False
        
    
        self.hybrid_encoder = HybridEncoder(num_classes, hidden_dim, hdc_dim, device)
        
       
        self.transformer_encoder = BertModel.from_pretrained('bert-base-uncased')
        
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, images, labels):
        image_features = self.image_encoder(images).last_hidden_state
        
        
        label_embeddings = self.hybrid_encoder(labels)
        
        
        combined_input = torch.cat([image_features, label_embeddings.unsqueeze(1)], dim=1)
        
        
        encoder_output = self.transformer_encoder(inputs_embeds=combined_input).last_hidden_state
        
        logits = self.classifier(encoder_output[:, 0, :])  
        
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


        enhanced_prototypes = prototypes + self.hybrid_encoder(support_labels[:, 0])

        dists = torch.cdist(query_features, enhanced_prototypes)

        return -dists

    def load_state_dict(self, state_dict):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name not in own_state:
                continue
            if isinstance(param, nn.Parameter):
                param = param.data
            own_state[name].copy_(param)
    
    def get_regularization_loss(self):
        l2_loss = torch.sum(self.classifier.weight ** 2)
        
        for name, param in self.transformer_encoder.named_parameters():
            if 'weight' in name:
                l2_loss += torch.sum(param ** 2)
        
        l2_loss += torch.sum(self.hybrid_encoder.elmes_scale ** 2)
        
        return l2_loss

