import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_chm(image_features, text_features, image_pseudo_labels, margin, n_iter):
    # normalized features
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    batch_size = image_features.shape[0]

    batch_image_pseudo_labels = torch.tensor( image_pseudo_labels[n_iter * batch_size: (n_iter + 1) * batch_size] )

    image_mask = (batch_image_pseudo_labels.unsqueeze(0) == batch_image_pseudo_labels.unsqueeze(1)) & (batch_image_pseudo_labels.unsqueeze(0) != -1)
    image_labels = torch.where(image_mask, torch.tensor(1.0), torch.tensor(0.0))
    image_true_label = image_labels.fill_diagonal_(1)

    image_true_label = image_true_label.to(image_features.device)

    similarity_scores1 = torch.matmul(text_norm, image_norm.t())    
    similarity_scores2 = torch.matmul(image_norm, text_norm.t())  

    positive_pair_score1 = torch.diag(similarity_scores1)
    positive_pair_score2 = torch.diag(similarity_scores2)

    negative_similarity_scores1 = torch.where(image_true_label == 0, similarity_scores1, torch.tensor(-1.0))
    negative_similarity_scores2 = torch.where(image_true_label == 0, similarity_scores2, torch.tensor(-1.0))

    negative_pair_score1,_ = negative_similarity_scores1.topk(1, dim=1, largest=True, sorted=True)   
    negative_pair_score1 = negative_pair_score1.flatten()   

    negative_pair_score2,_ = negative_similarity_scores2.topk(1, dim=1, largest=True, sorted=True)  
    negative_pair_score2 = negative_pair_score2.flatten()   

    loss1 = torch.sum(torch.clamp(negative_pair_score1 - positive_pair_score1 + margin, min=0))
    loss2 = torch.sum(torch.clamp(negative_pair_score2 - positive_pair_score2 + margin, min=0))

    loss =  (loss1 + loss2) / 2

    return loss

def compute_cdm(image_features, text_features, image_pseudo_labels, n_iter, logit_scale, epsilon=1e-8):
    # normalized features
    image_norm = image_features / image_features.norm(dim=1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=1, keepdim=True)

    t2i_dot_product = text_norm @ image_norm.t()
    i2t_dot_product = t2i_dot_product.t()

    text_proj_image = logit_scale * t2i_dot_product
    image_proj_text = logit_scale * i2t_dot_product

    batch_size = image_features.shape[0]

    image_labels_select = torch.tensor( image_pseudo_labels[n_iter * batch_size: (n_iter + 1) * batch_size] )

    image_pseudo_pid = image_labels_select.reshape((batch_size, 1))    # make sure image_pseudo_pid size is [64, 1]
    image_pid_dist = image_pseudo_pid - image_pseudo_pid.t()    
    image_labels = (image_pid_dist == 0).float()    

    # normalize the true matching distribution
    image_labels_distribute = image_labels / image_labels.sum(dim=1)

    image_labels_distribute = image_labels_distribute.to("cuda")

    i2t_pred = F.softmax(image_proj_text, dim=1)
    i2t_loss = i2t_pred * ( F.log_softmax(image_proj_text, dim=1) - torch.log(image_labels_distribute + epsilon) )  
    
    t2i_pred = F.softmax(text_proj_image, dim=1)
    t2i_loss = t2i_pred * (F.log_softmax(text_proj_image, dim=1) - torch.log(image_labels_distribute + epsilon))

    loss = torch.mean(torch.sum(i2t_loss, dim=1)) + torch.mean(torch.sum(t2i_loss, dim=1))

    return loss



def compute_itc(image_features, text_features, logit_scale):
    """
    image-text contrastive (ITC) loss, InfoNCE
    """
    batch_size = image_features.shape[0]
    labels = torch.arange(start=0, end=batch_size, dtype=torch.int64)
    labels = labels.to(image_features.device)

    # normalized features
    image_norm = image_features / image_features.norm(dim=-1, keepdim=True)
    text_norm = text_features / text_features.norm(dim=-1, keepdim=True)

    # cosine similarity as logits
    logits_per_image = logit_scale * image_norm @ text_norm.t()
    logits_per_text = logits_per_image.t()

    loss_i = F.cross_entropy(logits_per_image, labels)
    loss_t = F.cross_entropy(logits_per_text, labels)
    loss = (loss_i +  loss_t)/2

    return loss

