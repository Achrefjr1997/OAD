from model import OADModel,OaDataset,transformer,OaDPrediction,list_checkpoints
from utils import download_files,prepare_data,normalize_sentinel2_image,AoDLoss
import os
import torch
import argparse
import pandas as pd
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
import torch.nn as nn
from tqdm import tqdm
def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="OAD Training", add_help=add_help)

    parser.add_argument("--data-path", default="Dataset/test_images", type=str, help="dataset path")
    parser.add_argument("--device", default="cpu", type=str, help="device (Use cuda or cpu Default: cuda)")
    parser.add_argument("-b", "--batch-size", default=1, type=int, help="images per gpu, the total batch size is $NGPU x batch_size")
    parser.add_argument("--models", default="OADmodels", type=str, metavar="N", help="models folder")
    parser.add_argument("--output", default="predictions", type=str, metavar="N", help="output folder")
    
    return parser



def main(args):
    root=args.data_path
    device=args.device
    batch_size=args.batch_size
    models=args.models
    output=args.output
    checkpoint_paths = list_checkpoints(models, extension='.pt')  
    ds=OaDPrediction(root)
    os.makedirs(output, exist_ok=True)
    dl = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=0
    )
    i=0
    list_out=[]
    # Iterate over each fold
    for fold, checkpoint_path in enumerate(checkpoint_paths, 1):
        model = OADModel(1)
        state_dict = torch.load(checkpoint_path)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        filenames = []
        predictions = []
        i=i+1
        # Iterate through the dataloader
        with tqdm(dl, leave=True) as pbar:
            for x, img_filename in pbar:
                x = x.to(device)
                with torch.no_grad():  # No gradient computation needed for inference
                    y = model(x)
                # Collect predictions and filenames
                filenames.append(img_filename[0])  # Assuming img_filename is a list with one item
                predictions.append(y.cpu().numpy().flatten())  # Flatten to a 1D array if needed

        # Convert lists to a DataFrame
        df_results = pd.DataFrame({
            'filename': filenames,
            'prediction': [float(pred[0]) for pred in predictions]  # Assuming single value prediction
        })

        # Save DataFrame to CSV
        output_csv_path = f'{output}/predictions_fold_{i}_{fold}.csv'
        list_out.append(output_csv_path)
        df_results.to_csv(output_csv_path, index=False, header=False)

        print(f'{checkpoint_path} to {output_csv_path}')
    
    dfs = [pd.read_csv(file, header=None, names=['filename', 'prediction']) for file in list_out]

    # Merge all DataFrames on the 'filename' column
    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = pd.merge(merged_df, df, on='filename', suffixes=('', '_drop'))

    # Select only the 'filename' column once, and all the 'prediction' columns
    prediction_columns = [col for col in merged_df.columns if 'prediction' in col]

    # Calculate the average of the predictions
    merged_df['average_prediction'] = merged_df[prediction_columns].mean(axis=1)

    # Keep only the 'filename' and 'average_prediction' columns
    final_df = merged_df[['filename', 'average_prediction']]
    final_df.to_csv(f'{output}/average_predictions.csv', index=False, header=False)
    print(f'results saved to output folder {output}')  




if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)