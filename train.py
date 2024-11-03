from model import OADModel,OaDataset,transformer
from utils import download_files,prepare_data,normalize_sentinel2_image,AoDLoss
import os
import torch
import argparse
import pandas as pd
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="OAD Training", add_help=add_help)

    parser.add_argument("--data-path", default="/Dataset/", type=str, help="dataset path")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=32, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--epochs", default=50, type=int, metavar="N", help="number of total epochs to run")
    parser.add_argument("--usePCA", action='store_true', help="Using PCA for image processing")
    parser.add_argument("--useIndexes", action='store_true', help="Using Indexes for image processing")
    parser.add_argument("--Folder", default=5, type=int, metavar="N", help="number of folders to run")
    parser.add_argument("--output", default="OADmodels", type=str, metavar="N", help="output folder")
    parser.add_argument("--earlystop", default=10, type=int, metavar="N", help="EARLY STOPPING EPOCH")
    parser.add_argument("--lr", default=0.0002, type=float, metavar="LR", help="initial learning rate", dest="lr")
    
    return parser



def main(args):
    root=args.data_path
    device=args.device
    batch_size=args.batch_size
    epochs=args.epochs

    usePCA=args.usePCA
    useIndexes=args.useIndexes
    Folder=args.Folder
    output=args.output
    earlystop=args.earlystop
    lr=args.lr
    WD = 1e-2
    print(f"Preparing data with ACP3C={usePCA}, include_index={useIndexes} ..........................")
    prepare_data.prepare_data(ACP3C=False, include_index=False)
    print(f"data prepared" )

    os.makedirs(output, exist_ok=True)
    data = pd.read_csv('Processed/train_answer.csv', header=None, names=['File_Name', 'Region', 'Value'])
    print(data.head())

    train_data = pd.DataFrame()
    test_data = pd.DataFrame()
    for region, group in data.groupby('Region'):
        # Sample 20% for testing
        test_sample = group.sample(frac=0.2, random_state=42)  # random_state for reproducibility
        train_sample = group.drop(test_sample.index)  # The rest goes to training
        
        # Append to train and test DataFrames
        train_data = pd.concat([train_data, train_sample])
        test_data = pd.concat([test_data, test_sample])
    GRAD_ACC=1
    N_WORKERS = os.cpu_count() 
    skf = KFold(n_splits=Folder, shuffle=True)
    for fold, (trn_idx, val_idx) in enumerate(skf.split(range(len(train_data)))):
        print(len(train_data), len(test_data))
        train_ds= OaDataset(root="Processed", df=train_data, imsz=(224, 224),transform=transformer(0.95))
        valid_ds= OaDataset(root="Processed", df=train_data, imsz=(224, 224),transform=None)
        
        train_dl = DataLoader(
            train_ds,
            batch_size=batch_size,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS
        )
        valid_dl = DataLoader(
            valid_ds,
            batch_size=batch_size//2,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
            num_workers=N_WORKERS
        )

        model = OADModel(1)
    #     print(model)
    #     print(model)²+²²

        model.to(device)
        
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=WD)

        warmup_steps = epochs / 10 * len(train_dl) 
        num_total_steps = epochs * len(train_dl) 
        num_cycles = 0.475
        scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=warmup_steps,
                                                    num_training_steps=num_total_steps,
                                                    num_cycles=num_cycles)

    #     criterion = nn.MSELoss()
        criterion = nn.L1Loss()  # Best performing loss
        criterion1 = AoDLoss()

        best_loss = float('inf')
        best_metric = -float('inf')
        best_train_metric = -float('inf')  # Initialize best training metric
        es_step = 0

        for epoch in range(1, epochs + 1):
            print(f'Start epoch {epoch}')
            model.train()
            total_loss = 0
            total_metric = 0  # Initialize metric for training
            with tqdm(train_dl, leave=True) as pbar:
                optimizer.zero_grad()
                for idx, (x, t) in enumerate(pbar):
                    x = x.to(device)
                    t = t.to(device)
                    
                    y = model(x)
                    loss = criterion(y.float(), t.unsqueeze(1).float())
                    total_loss += loss.item()

                    # Compute training metric using AoDLoss or another metric
                    metric = criterion1(y.float(), t.unsqueeze(1).float())
                    total_metric += metric.item()

                    if not math.isfinite(loss):
                        print(f"Loss is {loss}, stopping training")
                        sys.exit(1)
        
                    pbar.set_postfix(
                        OrderedDict(
                            loss=f'{loss.item() * GRAD_ACC:.6f}',
                            metric=f'{metric.item():.6f}',  # Display the metric for each batch
                            lr=f'{optimizer.param_groups[0]["lr"]:.3e}'
                        )
                    )
                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(model.parameters(), MAX_GRAD_NORM or 1e9)
                    
                    if (idx + 1) % GRAD_ACC == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                        if scheduler is not None:
                            scheduler.step()                    
        
            train_loss = total_loss / len(train_dl)
            train_metric = total_metric / len(train_dl)  # Average metric over all batches
            print(f'train_loss: {train_loss:.6f}, train_metric: {train_metric:.6f}')
    
            # Check if current training metric is the best
            if train_metric > best_train_metric:
                best_train_metric = train_metric
                print(f'New best training metric: {train_metric:.6f}, saving model.')
                
                fname= f'{output}/best_train_model_fold-{fold}.pt'
                torch.save(model.state_dict(), fname)

            # Validation
            total_metric = 0
            model.eval()
            with tqdm(valid_dl, leave=True) as pbar:
                with torch.no_grad():
                    for idx, (x, t) in enumerate(pbar):
                        x = x.to(device)
                        t = t.to(device)
                        y = model(x)
                        metric = criterion1(y.float(), t.unsqueeze(1).float())
                        total_metric += metric.item()   
        
            val_metric = total_metric / len(valid_dl)
            
            print(f'val_loss: {val_metric:.6f}')

            if val_metric > best_metric:
                es_step = 0

                if device != 'cuda:0':
                    model.to('cuda:0')                
                    
                print(f'epoch: {epoch}, best metric updated from {best_metric:.6f} to {val_metric:.6f}')
                best_metric = val_metric
                fname = f'{output}/best_wll_model_fold-{fold}.pt'
                torch.save(model.state_dict(), fname)
                
                if device != 'cuda:0':
                    model.to(device)
                
            else:
                es_step += 1
                if es_step >= earlystop:
                    print('Early stopping')
                    break




if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)