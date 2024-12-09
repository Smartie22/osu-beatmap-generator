'''
This module will be where all training is structured and done. Will import from the other modules.
'''
import json
import os

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from preprocessing import collate_batch_selector, BeatmapDataset
import preprocessing
from step_selector import StepSelectorEncoder, StepSelectorDecoder


def get_accuracy(encoder, decoder, dataset, max_samples=1000):
    """
    Calculate the accuracy of our model
    """
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn=collate_batch_selector)
    num_total = 0
    num_correct = 0
    pad_idx = 3
    for i, (x, t) in enumerate(dataloader):
        if num_total > max_samples:
            break
        # x represents the tokenized time-stamp input sequence, 
        # t represents the tokenized target hitobject sequence

        encoder_out, encoder_hd = encoder(x)                          #TODO: test
        decoder_out, _, _ = decoder(encoder_out, encoder_hd) #TODO: test
        
        #lossTODO: determine the accuracy across all tokens generated and their respective targets
        #notes: decoder_out is (1, L), where L is the length of the longest sequence in the batch
        
        #NOTE: decoder_out is a list containing:
        #   list of logits
        #i.e. each element in decoder_out is a list of probabilities
        
        for target, pred_logits in zip(t.squeeze(0), decoder_out):
            predicted = torch.stack(pred_logits).argmax(dim=-1)
            if target == pad_idx:
                break 
            if predicted == target:
                num_correct += 1
            num_total += 1

    #return accuracy
    return num_correct / num_total if num_total > 0 else 0
        
def train_selector(encoder, 
                   decoder, 
                   train_data, 
                   val_data, 
                   learning_rate = 0.001, 
                   batch_size = 100, 
                   num_epochs = 10, 
                   plot_every = 50, 
                   plot = True):
      
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=preprocessing.collate_batch_selector, shuffle=True) 
    criteron = torch.nn.CrossEntropyLoss()
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    iters, train_loss, train_acc, val_acc = [], [], [], [] 
    iter_count = 0
    try:
        for e in range(num_epochs):
            for i, (X, t) in enumerate(train_loader):
                optimizer_enc.zero_grad() #clean up accumulated gradients before any calculations
                optimizer_dec.zero_grad()
                
                #produce sequences of logits
                e_hd, e_out = encoder(X)
                d_out, _, _ = decoder(e_out, e_hd) #idk if we need the decoder final hidden layer

                # d_out_tensor = tensor([    # t_flatten = tensor([0, 2, 0, 2]) 
                #   [2.5, 1.2, 0.3],         # t is target index for each list
                #   [0.2, 1.1, 2.8],
                #   [1.5, 0.7, 0.6],
                #   [0.3, 1.5, 2.1]
                # ])

                # softmax(d_out_tensor) = tensor([
                #   [0.7859, 0.1749, 0.0392],  # softmax of [2.5, 1.2, 0.3]
                #   [0.0780, 0.2121, 0.7099],  # softmax of [0.2, 1.1, 2.8]
                #   [0.6590, 0.2415, 0.0995],  # softmax of [1.5, 0.7, 0.6]
                #   [0.0970, 0.2626, 0.6404]   # softmax of [0.3, 1.5, 2.1]
                # ])
                d_out_tensor = torch.cat([torch.stack(logits) for logits in d_out], dim=0)
                t_tensor = t.view(-1)
                loss = criteron(d_out_tensor, t_tensor)

                loss.backward() #propogate gradients
                optimizer_enc.step() #update params
                optimizer_dec.step()
                
                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = get_accuracy(encoder, decoder, train_data)
                    va = get_accuracy(encoder, decoder, val_data)
                    train_loss.append(loss)
                    train_acc.append(ta)
                    val_acc.append(va)
                    print("Iteration", iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
                    pass
    finally:
        if plot:
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.legend(["Train", "Validation"])

def train_models():
    pass


if __name__ == "__main__":
    dir = os.path.dirname(__file__)
    path = os.path.join(dir, '..', 'data')

    n_buckets = 10000
    path_tok_ind_e = os.path.join(dir, "test_encoder_tokens_to_idx.json")
    path_ind_tok_e = os.path.join(dir, "test_encoder_idx_to_token.json")
    path_tok_ind_d = os.path.join(dir, "test_tokenizer.json")
    path_ind_tok_d = os.path.join(dir, "test_indices.json")

    if not os.path.exists(path_tok_ind_e) or not os.path.exists(path_ind_tok_e):
        preprocessing.create_tokens_encoder(path_tok_ind_e, path_ind_tok_e, n_buckets)
    if not os.path.exists(path_tok_ind_d) or not os.path.exists(path_ind_tok_d):
        preprocessing.create_tokens_decoder(path, path_tok_ind_d, path_ind_tok_d)

    # open the files here
    print("Opening Files")
    fd_tok_ind_e = open(path_tok_ind_e,)
    fd_ind_tok_e = open(path_ind_tok_e,)
    fd_tok_ind_d = open(path_tok_ind_d,)
    fd_ind_tok_d = open(path_ind_tok_d,)
    tok_ind_e = json.load(fd_tok_ind_e)
    ind_tok_e = json.load(fd_ind_tok_e)
    tok_ind_d = json.load(fd_tok_ind_d)
    ind_tok_d = json.load(fd_ind_tok_d)

    # Create datasets
    print("Creating Datasets") # TODO: Some errors here about the BeatmapDataset list indices must be integers or slices, not list.
    n_dpoints = 10000
    bm = BeatmapDataset(path, tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d, n_buckets, n_dpoints)
    train_set, val_set = torch.utils.data.random_split(bm, [len(bm) - 2000, 2000])
    test_set = val_set[1000:]
    val_set = val_set[:1000]

    # Create models
    print("Creating Models")
    emb_size = 200
    hidden_size_e = 200
    enc = StepSelectorEncoder(n_buckets, emb_size, hidden_size_e)
    hidden_size_d = 300
    dec = StepSelectorDecoder(len(ind_tok_d.keys()), hidden_size_d)

    # Train models
    print("Training Models")
    train_selector(enc, dec, train_set, val_set)

    fd_tok_ind_e.close()
    fd_ind_tok_e.close()
    fd_tok_ind_d.close()
    fd_ind_tok_d.close()