from sklearn.metrics.pairwise import cosine_similarity
import torch
from tqdm.auto import tqdm
import numpy as np

def train(data_loader, model, criterion, optimizer, device):
    model.train()
    epoch_losses = []
    epoch_sim = []
    epoch_acc = []
    loss_2 = []
    for batch in tqdm(data_loader, desc="training..."):
        #Load input
        mean_ids = batch["mean_query"].to(device)
        label = batch["rewrite"].to(device)
        ids = batch["embed_query"].to(device)
        mean_ctx = batch["mean_context"].to(device)
        ctx = batch["context"].to(device)
        # label_pass = batch["label_passage"].to(device)
        # passage = batch["batch_pass"].to(device)
        batch_irrelevant = batch["batch_irrelevant"].to(device)
        # Run model
        output = model(mean_ids,ids, mean_ctx, ctx)
        # Calculate loss
        loss = criterion(output, label, batch["batch_relevant"], batch_irrelevant)
        # Score
        similarity_score = cosine_similarity(output.reshape(1, -1).detach().cpu().numpy(),label.reshape(1, -1).detach().cpu().numpy())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_losses.append(loss.item())
        epoch_sim.append(similarity_score)
    return np.mean(epoch_losses), np.mean(epoch_sim)

def evaluate(data_loader, model, criterion,device):
    model.eval()
    epoch_losses = []
    epoch_sim = []
    epoch_acc = []
    loss_2 = []
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="evaluating..."):
            #Load input
            mean_ids = batch["mean_query"].to(device)
            label = batch["rewrite"].to(device)
            ids = batch["embed_query"].to(device)
            mean_ctx = batch["mean_context"].to(device)
            ctx = batch["context"].to(device)
            # label_pass = batch["label_passage"].to(device)
            # passage = batch["batch_pass"].to(device)
            batch_irrelevant = batch["batch_irrelevant"].to(device)
            # Run model
            output = model(mean_ids,ids, mean_ctx, ctx)
            # Calculate score
            similarity_score = cosine_similarity(output.reshape(1, -1).detach().cpu().numpy(),label.reshape(1, -1).detach().cpu().numpy())
            # Check loss
            loss =  criterion(output, label, batch["batch_relevant"], batch_irrelevant)

            # cls = torch.diagonal(torch.Tensor(cosine_similarity(mean_ids.detach().cpu().numpy(),label.detach().cpu().numpy())).cuda(),0)
            # cls = (cls>0.95).float()
            # loss2 = criterion2(model.cls_output.squeeze(), cls)
            # accuracy = get_accuracy(model.cls_output.squeeze(), cls)
            # loss = loss*0.8 + loss2*0.2
            # epoch_acc.append(accuracy.cpu().numpy())
            # loss_2.append(loss2.item())

            epoch_losses.append(loss.item())
            epoch_sim.append(similarity_score)

    return np.mean(epoch_losses), np.mean(epoch_sim)
    # return np.mean(epoch_losses), np.mean(epoch_sim), np.mean(epoch_acc), np.mean(loss_2)

def get_accuracy(prediction, label):

    predicted_classes = (prediction>0.90).float()
    label_classes = label
    correct_predictions = predicted_classes.eq(label_classes).sum()
    accuracy = correct_predictions / prediction.shape[0]
    return accuracy

class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss, model):
        torch.save(model.state_dict(), "merge_emb_last.pt")
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
            torch.save(model.state_dict(), "merge_emb.pt")

        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False