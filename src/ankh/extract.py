import argparse
import pathlib
from ankh.models import ankh_transformers
from ankh import utils
import torch
from tqdm.auto import tqdm
import numpy as np


def create_parser() -> argparse.ArgumentParser:
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--model", type=str, help="Whether to use the base model or the large model."
    )
    argparser.add_argument("--fasta_path", type=str, help="Location to the fasta file.")
    argparser.add_argument(
        "--output_path", type=str, help="Location to save the embeddings."
    )
    argparser.add_argument("--use_gpu", type=bool, help="Whether to use GPU or not.")
    return argparser


def validate_output_path(path: pathlib.Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"File not found. Recieved path: {path}")


def get_device(use_gpu):
    if torch.cuda.is_available() and use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    return device


def main(args: argparse.Namespace) -> None:
    model, tokenizer = ankh_transformers.load_model(args.model)
    fasta_dataset = utils.FastaDataset(args.fasta_path)
    output_path = pathlib.Path(args.output_path)
    validate_output_path(output_path)

    device = get_device(args.use_gpu)

    shift_left = 0
    shift_right = -1

    model.eval()
    model.to(device=device)

    with torch.no_grad():
        for idx, sample in enumerate(tqdm(fasta_dataset), start=1):
            ids = tokenizer.batch_encode_plus(
                [sample],
                add_special_tokens=True,
                padding=True,
                is_split_into_words=True,
                return_tensors="pt",
            )
            embedding = model(input_ids=ids["input_ids"].to(device))[0]
            current_embeddings = embedding[0].cpu().numpy()[shift_left:shift_right]
            np.save(output_path / f"sequence_{idx}", current_embeddings)


if __name__ == "__main__":
    args = create_parser()
    main(args)
