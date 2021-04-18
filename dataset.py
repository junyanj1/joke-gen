import torch
import csv
import torchtext as tt


class JokeDataset(torch.utils.data.Dataset):

    def __init__(self, csv_file, max_seq_length):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
        """
        self.lines = []
        self.read_csv(csv_file, max_seq_length)


    def read_csv(self, file_path, max_seq_length):
        tokenizer = tt.data.utils.get_tokenizer("basic_english")

        with open(file_path, "r", encoding='utf8') as f:
            reader = csv.reader(f, delimiter=",")
            for line in reader:
                toks = tokenizer(line[1])
                if len(toks) < max_seq_length:
                    toks += ["<pad>"]*(max_seq_length-len(toks))
                if len(toks) > max_seq_length:
                    toks = toks[:max_seq_length]
                self.lines.append(toks)


    def __len__(self):
        return len(self.lines)

    def __getitem__(self, idx):
        return self.lines[idx]