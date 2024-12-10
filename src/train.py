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
    encoder.eval()
    decoder.eval()
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

        encoder_hd, encoder_out = encoder(x)                          #TODO: test
        decoder_out, _, _ = decoder(encoder_out, encoder_hd) #TODO: test
        
        #lossTODO: determine the accuracy across all tokens generated and their respective targets
        #notes: decoder_out is (1, L), where L is the length of the longest sequence in the batch
        
        #NOTE: decoder_out is a list containing:
        #   list of logits
        #i.e. each element in decoder_out is a list of probabilities
        
        for target, pred_logits in zip(t, decoder_out):
            predicted = pred_logits.argmax(dim=-1)
            # if target == pad_idx:
            #     break
            #print(predicted.shape, target.shape)
            #print(predicted)
            #print("target is", target)

            num_correct += int(torch.sum(target == predicted)) # TODO: Not sure if this is right
            # if predicted == target:
            #     num_correct += 1
            num_total += target.size(0)

    #return accuracy
    print("correct:", num_correct, "total:", num_total)
    return num_correct / num_total if num_total > 0 else 0
        
def train_selector(encoder, 
                   decoder, 
                   train_data, 
                   val_data, 
                   learning_rate = 0.001, 
                   batch_size = 10, 
                   num_epochs = 100, 
                   plot_every = 50, 
                   plot = True):
      
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=preprocessing.collate_batch_selector, shuffle=True) 
    criteron = torch.nn.CrossEntropyLoss()
    optimizer_enc = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    optimizer_dec = torch.optim.Adam(decoder.parameters(), lr=learning_rate)

    iters, train_loss, train_acc, val_acc = [], [], [], [] 
    iter_count = 0
    try:
        print("---------------------------------------------------------------------------\nbeginning training\nbatchsize is", batch_size, "num_epochs is", num_epochs, "we plot every", plot_every, "data points\n---------------------------------------------------------------------------\n")
        encoder.train()
        decoder.train()
        for e in range(num_epochs):
            for i, (X, t) in enumerate(train_loader):
                print("training loop:", i, "epoch:", e)
                optimizer_enc.zero_grad() #clean up accumulated gradients before any calculations
                optimizer_dec.zero_grad()
                
                #produce sequences of logits
                e_hd, e_out = encoder(X)
                d_out, _, _ = decoder(e_out, e_hd) #idk if we need the decoder final hidden layer
                d_out_tensor = d_out.transpose(1, 2)

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
                # d_out_tensor = torch.cat([torch.stack(logits) for logits in d_out], dim=0)
                t_tensor = t
                loss = criteron(d_out_tensor, t_tensor)
#                loss.requires_grad = True
                loss.backward() #propogate gradients
                torch.nn.utils.clip_grad_value_(encoder.parameters(), 0.8)
                torch.nn.utils.clip_grad_value_(decoder.parameters(), 0.8)
                optimizer_enc.step() #update params
                optimizer_dec.step()
                
                iter_count += 1
                if iter_count % plot_every == 0:
                    #print("calculating accuracy")
                    iters.append(iter_count)
                    ta = get_accuracy(encoder, decoder, train_data)
                    va = get_accuracy(encoder, decoder, val_data)
                    encoder.train()
                    decoder.train()
                    train_loss.append(loss.item())
                    train_acc.append(ta)
                    val_acc.append(va)
                    print("Iteration", iter_count, "Loss:", float(loss), "Train Acc:", ta, "Val Acc:", va)
    except e:
        print("ERROR !!!!!")
        print(e)
    finally:
        if plot:
            print("plotting")
            plt.figure()
            plt.plot(iters[:len(train_loss)], train_loss)
            plt.title("Loss over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Loss")
            plt.savefig("loss_over_iters.png")

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Acc")
            plt.legend(["Train", "Validation"])
            plt.savefig("acc_over_iters.png")


def create_vocab_open_token_files(dir, path, n_buckets):
    print("Creating vocab")
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

    #close files
    fd_tok_ind_e.close()
    fd_ind_tok_e.close()
    fd_tok_ind_d.close()
    fd_ind_tok_d.close()


    #return
    return tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d

def create_datasets(path, tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d, n_dpoints, n_buckets):
    print("Creating Datasets")
    bm = BeatmapDataset(path, tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d, n_buckets, n_dpoints)
    train_set = bm.data[:24]
    val_set = bm.data[24:]
    test_set = val_set[0]
    return train_set, val_set, test_set

def create_models(n_buckets, emb_size, hidden_size_e, hidden_size_d, output_size_d):
    print("Creating Models")
    enc = StepSelectorEncoder(n_buckets, emb_size, hidden_size_e)
    dec = StepSelectorDecoder(output_size_d, hidden_size_d)
    #write these hyper parameter values out to be recreated
    write_hyperparams(n_buckets, emb_size, hidden_size_e, hidden_size_d, output_size_d)
    return enc, dec

def write_hyperparams(n_buckets, emb_size, hidden_size_e, hidden_size_d, output_size_d):
    currpath = os.path.dirname(__file__) #current file path
    with open(os.path.join(currpath, 'hyper-params-selector')) as outfile:
        hyperparams = {}
        hyperparams['n_buckets'] = n_buckets 
        hyperparams['emb_size'] = emb_size
        hyperparams['hidden_size_e'] = hidden_size_e
        hyperparams['hidden_size_d'] = hidden_size_d
        hyperparams['output_size_d'] = output_size_d
        json.dump(hyperparams, outfile)


def set_up_and_train():  
    dir = os.path.dirname(__file__)
    datapath = os.path.join(dir, '..', 'data')
    n_buckets = 10000
    tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d = create_vocab_open_token_files(dir, datapath, n_buckets)

    # Create datasets
    n_dpoints = 10000
    train_set, val_set, test_set = create_datasets(datapath, tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d, n_dpoints, n_buckets)


    # Create models
    emb_size = 200
    hidden_size_e = 200
    hidden_size_d = hidden_size_e
    output_size_d = len(tok_ind_d.keys()) #output size is the number of decoder tokens possible
    enc, dec = create_models(n_buckets, emb_size, hidden_size_e, hidden_size_d, output_size_d)

    # Train models
    print("Training Models")
    train_selector(enc, dec, train_set, val_set, num_epochs=1, plot_every=1)

    print("done, exporting models to './encoder.pt' and './decoder.pt'")
    torch.save(enc.state_dict(), os.path.join(dir, 'encoder.pt'))
    torch.save(dec.state_dict(), os.path.join(dir, 'decoder.pt'))



if __name__ == "__main__":
    set_up_and_train()