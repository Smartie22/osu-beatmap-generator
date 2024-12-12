'''
This module will be where all training is structured and done. Will import from the other modules.
'''
import json
import os
import shutil

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

        encoder_hd, encoder_out = encoder(x)
        decoder_out, _, _ = decoder(encoder_out, encoder_hd)
        #notes: decoder_out is (1, L), where L is the length of the longest sequence in the batch
        #NOTE: decoder_out is a list containing:
        #   list of logits

        #i.e. each element in decoder_out is a list of probabilities
        for target, pred_logits in zip(t, decoder_out):
            predicted = pred_logits.argmax(dim=-1)
            num_correct += int(torch.sum(target == predicted))
            num_total += target.size(0)

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
                   plot = True,
                   acc_graph_path="acc_over_iters.png",
                   loss_graph_path="loss_over_iters.png",
                   param_path = None
                   ):
      
    train_loader = DataLoader(train_data, batch_size=batch_size, collate_fn=preprocessing.collate_batch_selector, shuffle=True) 
    criteron = torch.nn.CrossEntropyLoss(ignore_index=2)
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
                if param_path is not None:
                    with open(param_path, "a") as outfile:
                        outfile.write(f"training loop: {i} epoch: {e}\n")
                print("training loop:", i, "epoch:", e)
                optimizer_enc.zero_grad() #clean up accumulated gradients before any calculations
                optimizer_dec.zero_grad()
                
                #produce sequences of logits
                e_hd, e_out = encoder(X)
                d_out, _, _ = decoder(e_out, e_hd, t) #idk if we need the decoder final hidden layer

                d_out_tensor = d_out.transpose(0, 1).squeeze()
                d_out_tensor = d_out_tensor.transpose(1, 2)
                t_tensor = t
                loss = criteron(d_out_tensor, t_tensor)
                loss.backward() #propogate gradients
                torch.nn.utils.clip_grad_value_(encoder.parameters(), 0.8)
                torch.nn.utils.clip_grad_value_(decoder.parameters(), 0.8)
                optimizer_enc.step() #update params
                optimizer_dec.step()
                
                iter_count += 1
                if iter_count % plot_every == 0:
                    iters.append(iter_count)
                    ta = get_accuracy(encoder, decoder, train_data)
                    va = get_accuracy(encoder, decoder, val_data)
                    encoder.train()
                    decoder.train()
                    train_loss.append(loss.item())
                    train_acc.append(ta)
                    val_acc.append(va)
                    if param_path is not None:
                        with open(param_path, "a") as outfile:
                            outfile.write(f"Iteration: {iter_count} Loss: {float(loss)} Train Acc: {ta} Val Acc: {va}\n")
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
            plt.savefig(loss_graph_path)

            plt.figure()
            plt.plot(iters[:len(train_acc)], train_acc)
            plt.plot(iters[:len(val_acc)], val_acc)
            plt.title("Accuracy over iterations")
            plt.xlabel("Iterations")
            plt.ylabel("Acc")
            plt.legend(["Train", "Validation"])
            plt.savefig(acc_graph_path)


def create_vocab_open_token_files(dir, path, n_buckets):
    print("Creating vocab")
    path_tok_ind_e = os.path.join(dir, "test_encoder_tokens_to_idx.json")
    path_ind_tok_e = os.path.join(dir, "test_encoder_idx_to_token.json")
    path_tok_ind_d = os.path.join(dir, "test_tokenizer.json")
    path_ind_tok_d = os.path.join(dir, "test_indices.json")

    if os.path.exists(path_tok_ind_e) or os.path.exists(path_ind_tok_e):
        os.remove(path_tok_ind_e)
        os.remove(path_ind_tok_e)
    if not (os.path.exists(path_tok_ind_d) or os.path.exists(path_ind_tok_d)):
        preprocessing.create_tokens_decoder(path, path_tok_ind_d, path_ind_tok_d)
    preprocessing.create_tokens_encoder(path_tok_ind_e, path_ind_tok_e, n_buckets)


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
    num_train = (int)(0.8*len(bm))
    train_set = bm.data[:num_train]
    val_set = bm.data[num_train:]
    test_set = []
    return train_set, val_set, test_set

def create_models(n_buckets, emb_size, hidden_size_e, hidden_size_d, output_size_d):
    print("Creating Models")
    enc = StepSelectorEncoder(n_buckets, emb_size, hidden_size_e)
    dec = StepSelectorDecoder(output_size_d, hidden_size_d)
    return enc, dec

def create_params_file(file_w_path, param_dict):
    with open(file_w_path, "w") as outfile:
        # Overwrite existing content with new things...
        for k in param_dict:
            outfile.write(f"{k}: {param_dict[k]}\n")
        outfile.write("\n")

def grid_search(num_epochs, plot_every):
    curr_dir = os.path.dirname(__file__)
    res_dir_path = os.path.join(curr_dir, "grid_search_results")
    if os.path.exists(res_dir_path):
        shutil.rmtree(res_dir_path)
    os.mkdir(res_dir_path)

    ne = num_epochs # num epochs. Please adjust during final testing. num epoch default: 50
    pe = plot_every # plot every iteration. Please adjust during final testing. plot every default: 20

    nb_lst = [1000, 10000] # num buckets
    emb_lst = [200, 300] # embedding size
    hs_lst = [200, 300] # hidden size
    bs_lst = [30, 40] # batch size

    for num_buckets in nb_lst:
        for emb in emb_lst:
            for hidden in hs_lst:
                for bs in bs_lst:
                    lr = 0.01
                    new_dir_name = f"nb_{num_buckets}_es_{emb}_hs_{hidden}_ne_{ne}_pe_{pe}_lr_{lr}_bs_{bs}"
                    new_dir_path = os.path.join(res_dir_path, new_dir_name)
                    os.mkdir(new_dir_path)
                    
                    param_dict = {}
                    param_dict['n_buckets'] = num_buckets
                    param_dict['emb_size'] = emb
                    param_dict['hidden_size_e'] = hidden
                    param_dict['hidden_size_d'] = hidden
                    param_dict['num_epoch'] = ne
                    param_dict['plot_every'] = pe
                    param_dict['learning_rate'] = lr
                    param_dict['batch_size'] = bs
                    param_dict['acc_graph_path'] = os.path.join(new_dir_path, "acc_graph.png")
                    param_dict['loss_graph_path'] = os.path.join(new_dir_path, "loss_graph.png")
                    param_dict['encoder_file_path'] = os.path.join(new_dir_path, "encoder.pt")
                    param_dict['decoder_file_path'] = os.path.join(new_dir_path, "decoder.pt")
                    file_path = os.path.join(new_dir_path, "results.txt")

                    set_up_and_train(file_path, param_dict)

def set_up_and_train(param_path=None, param_dict=None):
    dir = os.path.dirname(__file__)
    datapath = os.path.join(dir, '..', 'data')

    # Initialize all parameters
    n_buckets = 10000
    emb_size = 200
    hidden_size_e = 200
    hidden_size_d = hidden_size_e
    num_epoch = 1
    plot_every = 1
    learning_rate = 0.001
    batch_size = 10
    acc_graph_path = "acc_over_iters.png"
    loss_graph_path = "loss_over_iters.png"
    encoder_file_path = os.path.join(dir, 'encoder.pt')
    decoder_file_path = os.path.join(dir, 'decoder.pt')

    if param_dict is not None and param_path is not None:
        n_buckets = param_dict['n_buckets']
        emb_size = param_dict['emb_size']
        hidden_size_e = param_dict['hidden_size_e']
        hidden_size_d = param_dict['hidden_size_d']
        num_epoch = param_dict['num_epoch']
        plot_every = param_dict['plot_every']
        learning_rate = param_dict['learning_rate']
        batch_size = param_dict['batch_size']
        acc_graph_path = param_dict['acc_graph_path']
        loss_graph_path = param_dict['loss_graph_path']
        encoder_file_path = param_dict['encoder_file_path']
        decoder_file_path = param_dict['decoder_file_path']

        create_params_file(param_path, param_dict)
    else:
        param_path = None

    tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d = create_vocab_open_token_files(dir, datapath, n_buckets)

    # Create datasets
    n_dpoints = 10000
    train_set, val_set, test_set = create_datasets(datapath, tok_ind_e, ind_tok_e, tok_ind_d, ind_tok_d, n_dpoints, n_buckets)
    print("size of train set is", len(train_set), "size of val set is", len(val_set))

    # Create models
    output_size_d = len(tok_ind_d.keys()) #output size is the number of decoder tokens possible
    enc, dec = create_models(n_buckets, emb_size, hidden_size_e, hidden_size_d, output_size_d)

    # Train models
    print("Training Models")
    train_selector(enc, dec, train_set, val_set, learning_rate, batch_size, num_epoch, plot_every, acc_graph_path=acc_graph_path, loss_graph_path=loss_graph_path, param_path=param_path)

    print(f"done, exporting models to '{encoder_file_path}' and '{decoder_file_path}'\n")
    torch.save(enc.state_dict(), encoder_file_path)
    torch.save(dec.state_dict(), decoder_file_path)


if __name__ == "__main__":
    print("HEY! Be sure to delete the .json files before training or grid search if you have added more training data!!")
    grid_search(10, 5)
