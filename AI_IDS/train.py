import torch
import argparse
from tqdm import tqdm
from torch.nn import BCELoss
from torch.utils.data import DataLoader

from util import get_model, get_optimizer, get_dataset


def train(model, optim, dataloader, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    loss = BCELoss()
    model = model.to(device)

    for k in range(args.epoch):
        model.train()
        tbar = tqdm(dataloader)
        tot_loss = 0.0
        tbar.set_description(f'Epoch {k}')
        for x, y in tbar:
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            res = model(x)
            lss = loss(res, y)
            tot_loss += lss.detach().cpu().item()
            lss.backward()
            optim.step()
            tbar.set_postfix({'train loss': tot_loss})
    
        torch.save(model.state_dict(), args.save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # hyperparameter
    parser.add_argument('--model_type', default='dcnn', type=str, help='model type')
    parser.add_argument('--epoch', default=100, type=int, help='epoch')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--optimzer', default='adam', type=str, help='optimzer')
    parser.add_argument('--batch_size', default=64, type=int, help='batch size')
    parser.add_argument('--window_size', default=3, type=int, help='window size')
    
    # for lstm
    parser.add_argument('--hid_dim', default=512, type=int, help='hidden dimension')
    
    # else
    parser.add_argument('--save_path', default='./model.pth', type=str, help='model save path')
    parser.add_argument('--train_f_path', default='./train.csv', type=str, help='train file path')
    parser.add_argument('--test_f_path', default='./test.csv', type=str, help='test file path')
    
    args = parser.parse_args()
    model = get_model(args.model_type, args)
    optimizer = get_optimizer(args.optimizer, model, args.lr)
    train_data = get_dataset(args.train_f_path, args.model_type, args.window_size)
    train_dataloader = DataLoader(train_data, args.batch_size, True)
    train(model, optimizer, train_dataloader, args)
    