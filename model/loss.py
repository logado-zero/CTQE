import torch

class ContrastiveLoss(torch.nn.Module):
    # Basically the same as this: https://arxiv.org/pdf/2410.02525
    def __init__(self, alpha=0.8):
        super().__init__()
        self.alpha = alpha

    def cos_sim(self,a, b):
        # From https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py#L31
        """
        Computes the cosine similarity cos_sim(a[i], b[j]) for all i and j.
        :return: Matrix with res[i][j]  = cos_sim(a[i], b[j])
        """
        if not isinstance(a, torch.Tensor):
            a = torch.tensor(a)

        if not isinstance(b, torch.Tensor):
            b = torch.tensor(b)

        if len(a.shape) == 1:
            a = a.unsqueeze(0)

        if len(b.shape) == 1:
            b = b.unsqueeze(0)

        a_norm = torch.nn.functional.normalize(a, p=2, dim=1)
        b_norm = torch.nn.functional.normalize(b, p=2, dim=1)
        return torch.mm(a_norm, b_norm.transpose(0, 1))

    def forward(self, embeddings_src, embeddings_target, relevant_passage, irrelevant_passage):
        """
        Compute similarity between `a` and `b`.
        Labels have the index of the row number at each row.
        This indicates that `a_i` and `b_j` have high similarity
        when `i==j` and low similarity when `i!=j`.
        """

        scores =  self.cos_sim(embeddings_src, embeddings_target)
        #-------
        re_pos = []
        re_neg = []
        for i in range(embeddings_src.size(0)):
          # Calculate the logsumexp for positive examples
          if relevant_passage[i] is not None:
            passage = relevant_passage[i].to(embeddings_src.device)
            pos_score = 1 + torch.sum(torch.exp(self.cos_sim(embeddings_src[i], passage)))
          else:
            pos_score = torch.Tensor([1.])[0].to(embeddings_src.device)

          re_pos.append(pos_score)

          neg_score =  torch.sum(torch.exp(self.cos_sim(embeddings_src[i], irrelevant_passage[i])))
          re_neg.append(pos_score + neg_score)

        loss_pos = torch.log(torch.stack(re_pos))
        loss_neg = torch.log(torch.stack(re_neg))
        
        loss_main = torch.log(torch.exp(torch.diagonal(scores, 0)))

        loss = torch.mean( - (self.alpha*loss_main + (1-self.alpha)*(loss_pos - loss_neg)))
        return loss