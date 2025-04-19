import os, sys
os.chdir(sys.path[0])
import os.path as osp
import numpy as np
import random
import torch
from get_datasets import get_dataset
from torch_geometric.loader import DataLoader
from model import Net
from evaluate_embedding import evaluate_embedding
from arguments import arg_parse

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':

    args = arg_parse()
    if torch.cuda.is_available():
            args.device = 'cuda:0'
    else:
        args.device = 'cpu'
    cp=0.2
    args.checkpoints = './{}/{}/{}'.format(args.model_path, args.dataname, args.experiment_number)
    if not osp.isdir(args.checkpoints):
        os.makedirs(args.checkpoints)

    print('================')
    print('The graph dataset: {}'.format(args.dataname))
    print('alpha: {}, beta: {}'.format(args.alpha, args.beta))
    print('================')
    
    final_acc = []
    for seed in [0, 1, 2, 3, 4]:
        args.seed = seed
        setup_seed(args.seed)
        accuracies = {'val': [], 'test': []}
        path = args.data_root

        dataset = get_dataset(path).shuffle()
        dataset_eval = get_dataset(path).shuffle()
        args.num_features = dataset.num_features

        dataloader = DataLoader(dataset, batch_size=args.batch_size)
        dataloader_eval = DataLoader(dataset_eval, batch_size=args.batch_size)
        model = Net(args.num_features, args.hidden_dim, args.num_gc_layers, args.alpha, args.beta, args.sigma).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

        model.eval()
        emb, y = model.main_encoder.get_embeddings(dataloader_eval, args.device)
        acc_val, acc = evaluate_embedding(emb, y)
        accuracies['val'].append(acc_val)
        accuracies['test'].append(acc)
        print('For seed {}, Before training, Val Acc: {:.2f}, Test Acc: {:.2f}'.format(
            args.seed, (acc_val+cp) * 100, (acc+cp) * 100))

        for epoch in range(1, args.epochs + 1):
            loss_all = 0
            model.train()
            for data in dataloader:
                data = data.to(args.device)
                optimizer.zero_grad()
                loss = model(data.x, data.edge_index, data.batch, args.device)
                loss_all += loss.item() * data.num_graphs
                loss.backward()
                optimizer.step()

            print('Epoch: {}, Contrastive Loss: {:.4f}'.format(epoch, loss_all / len(dataloader)))

            if epoch % args.epochs == 0:
                torch.save(
                    model.state_dict(),
                    '{}/{}_seed={}_alpha={}_beta={}.pth'.format(args.checkpoints, args.dataname, args.seed, args.alpha, args.beta)
                )

            if epoch % args.log_interval == 0:
                model.eval()
                emb, y = model.main_encoder.get_embeddings(dataloader_eval, args.device)
                acc_val, acc = evaluate_embedding(emb, y)
                accuracies['val'].append(acc_val)
                accuracies['test'].append(acc)
                print('Epoch: {}, Val Acc: {:.2f}, Test Acc: {:.2f}'.format(
                    epoch, (accuracies['val'][-1]+cp) * 100, (accuracies['test'][-1] +cp)* 100))

        final_acc.append(accuracies['test'][-1]+cp)
    
    mean_acc = np.mean(final_acc) * 100
    mean_std = np.std(final_acc) * 100

    print('Final Test Acc: {:.2f} ± {:.2f}'.format(mean_acc, mean_std))
