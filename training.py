import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
import torch.optim as optim
import collections
from matplotlib import pyplot as plt
import os

from data.data_process import RetrieverDataset, get_collate_fn
from utils.help_function import initialize_weights, count_parameters, EarlyStopper, train, evaluate
from model.loss import ContrastiveLoss
from model.ctqe import CTQE


if __name__ == '__main__':
    result_path = "result/ctqe"
    os.makedirs(result_path, exist_ok=True)
    # Init training parameters
    BATCH_SIZE = 64
    dropout_rate = 0.25
    learning_rate = 1e-3

    n_layers = 2
    bidirectional = True
    num_heads = 8

    # Create DataLoader
    train_file_path = "data/raw/train/new_train_conversation.json"
    train_dataset_path = "data/processed/train_data_bertopic_all-MiniLM-L6-v2_negBM25.pkl"
    bertopic_context = "data/bertopic/BerTopic_corpus_all-MiniLM-L6-v2"
    vs_context_path = "data/vector_store_doc/faiss_CORAL_all-MiniLM-L6-v2"

    train_dataset = RetrieverDataset(train_file_path, history_num=2,bertopic_context_path=bertopic_context, pre_data_path=train_dataset_path,
                                     vs_context_path=vs_context_path)

    train_dataset.only_2 = False
    train_size = int(0.7 * len(train_dataset))
    valid_size = len(train_dataset) - train_size
    train_data, valid_data = torch.utils.data.random_split(train_dataset, [train_size, valid_size])

    train_sampler = RandomSampler(train_data)
    valid_sampler = SequentialSampler(valid_data)

    train_data_loader = DataLoader(
        train_data, sampler=train_sampler, batch_size=BATCH_SIZE, collate_fn=get_collate_fn(train_dataset.total_data.vs_context,train_dataset.total_data.bertopic,
                                                                                            tokenizer=train_dataset.tokenizer,
                                                                                            model_embedding=train_dataset.embedding_model,
                                                                                            ))
    valid_data_loader = DataLoader(
        valid_data, sampler=valid_sampler, batch_size=BATCH_SIZE, collate_fn=get_collate_fn(train_dataset.total_data.vs_context,train_dataset.total_data.bertopic,
                                                                                            tokenizer=train_dataset.tokenizer,
                                                                                            model_embedding=train_dataset.embedding_model,
                                                                                            ))


    # Init model
    lstm_param = {'n_layers': n_layers,
              'bidirectional':bidirectional}

    model = CTQE(
        train_dataset.embedding_model.config.hidden_size,
        dropout_rate,
        lstm_param,
        num_heads,
    )

    model.apply(initialize_weights)

    print(f"The model has {count_parameters(model):,} trainable parameters")

    # Set up model
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = optim.AdamW(model.parameters(),lr=learning_rate, betas=(0.9, 0.999), eps=1e-08,weight_decay=1e-06)
    criterion = ContrastiveLoss(alpha=0.8)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=1, last_epoch=-1, verbose=False)

    model = model.to(device)
    criterion = criterion.to(device)

    n_epochs = 26
    best_valid_loss = float("inf")

    metrics = collections.defaultdict(list)
    early_stopper = EarlyStopper(patience=4, min_delta=0.0001, save_path=result_path)

    # Training process
    for epoch in range(n_epochs):
        torch.autograd.set_detect_anomaly(True)
        train_loss, train_sim = train(
            train_data_loader, model, criterion,optimizer, device
        )
        scheduler.step()
        valid_loss, valid_sim = evaluate(valid_data_loader, model, criterion, device)
        metrics["train_losses"].append(train_loss)
        metrics["train_sim"].append(train_sim)
        metrics["valid_losses"].append(valid_loss)
        metrics["valid_sim"].append(valid_sim)
        print(f"epoch: {epoch}")
        print(f"train_loss: {train_loss:.5f}, train_sim: {train_sim:.5f}")
        print(f"valid_loss: {valid_loss:.5f}, valid_sim: {valid_sim:.5f}")

        if early_stopper.early_stop(valid_loss, model):
            break

    # Plot training and validation losses
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(metrics["train_losses"], label="train loss")
    ax.plot(metrics["valid_losses"], label="valid loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_xticks(range(n_epochs))
    ax.legend()
    ax.grid()
    plt.tight_layout()
    plt.savefig(os.path.join(result_path,"training_validation_loss.png"))  # Save the plot
    plt.show() 