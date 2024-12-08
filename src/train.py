'''
This module will be where all training is structured and done. Will import from the other modules.
'''

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from preprocessing import collate_batch_selector
import preprocessing


def get_accuracy(encoder, decoder, dataset, max=1000):
    """
    Calculate the accuracy of our model
    """
    dataloader = DataLoader(dataset,
                            batch_size=1,
                            collate_fn=collate_batch_selector)

    for i, (x, t) in enumerate(dataloader):
        # x represents the tokenized time-stamp input sequence, 
        # t represents the tokenized target hitobject sequence

        encoder_out, encoder_hd = encoder(x)                          #TODO: test
        decoder_out, decoder_hd, _ = decoder(encoder_out, encoder_hd) #TODO: test
        
        #lossTODO: determine the accuracy across all tokens generated and their respective targets
        #notes: decoder_out is (1, L), where L is the length of the longest sequence in the batch
        
        #NOTE: decoder_out is a list containing:
        #   list of logits
        #i.e. each element in decoder_out is a list of probabilities
        num_total = 0
        num_correct = 0
        pad_idx = 3
        while num_total < len(t) and num_total < len(decoder_out):
            _, prediction_idx = decoder_out[i].topk(1) #determine the index of the highest probability token
            if prediction_idx == pad_idx: #NOTE: idk if we should break upon finding a padding token (I think this is fine probably, unless our model generates a padding token in the middle of a map for some reason -seb)
                break 
            if prediction_idx == t[i]:
                num_correct += 1
            num_total += 1

        #return accuracy
        return num_correct / num_total 
        
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