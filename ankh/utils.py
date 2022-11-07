from torch.utils.data import Dataset
import pathlib


class FastaDataset(Dataset):
    def __init__(self, fasta_path):
        """
        A class for loading a FASTA file.
        Args:
            fasta_path: String specifying the path of the FASTA file
        """
        self.fasta_path = pathlib.Path(fasta_path)
        if not self.fasta_path.exists():
            raise FileNotFoundError(
                f"Fasta file does not exist. Recieved path: {fasta_path}."
            )

        self.sequences = []
        with open(self.fasta_path, "r") as ff:
            file = ff.readlines()

            for f in file:
                if not f.startswith(">"):
                    sequence = f.strip()
                    sequence = list(sequence)
                    self.sequences.append(sequence)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, index):
        return self.sequences[index]


class CSVDataset(Dataset):
    def __init__(self, dataframe, sequences_column_name, labels_column_name):
        """
        A class for loading a CSV file.
        Args:
            dataframe: `DataFrame` instance that contains the sequences and labels.
            sequences_column_name: String specifying the column name of the sequences.
            labels_column_name: String specifying the column name of the labels.
        """
        self.df = dataframe
        self.seq_col_name = sequences_column_name
        self.label_col_name = labels_column_name

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        return (
            list(self.df.iloc[index, self.seq_col_name]),
            self.df.iloc[index, self.label_col_name],
        )
