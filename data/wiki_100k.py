from .data_loader import Decoder, DataLoader
from huggingface_hub import hf_hub_download
import os

class WikiLoader:
    def __init__(self, config):
        self.config = config
        self.train_file_path = None
        self.val_file_path = None
        self.vocab_path = None

    def download_dataset(self):
        current_dir_path = os.path.dirname(os.path.realpath(__file__))

        os.mkdir('wiki_100k_tokenized') if not os.path.exists('wiki_100k_tokenized') else None
        download_dir = os.path.join(current_dir_path, 'wiki_100k_tokenized')

        hf_hub_download(repo_id="EzzeddineAlSai/wiki-100k-tokenized", repo_type="dataset", filename="validation.bin", local_dir=download_dir )
        hf_hub_download(repo_id="EzzeddineAlSai/wiki-100k-tokenized", repo_type="dataset", filename="train.bin", local_dir=download_dir )
        hf_hub_download(repo_id="EzzeddineAlSai/wiki-100k-tokenized", repo_type="dataset", filename="vocab.json", local_dir=download_dir )

        self.train_file_path = os.path.join(download_dir, 'train.bin')
        self.val_file_path = os.path.join(download_dir,'validation.bin')
        self.vocab_path = os.path.join(download_dir, 'vocab.json')

    def get_loaders_and_decoder(self):
        self.download_dataset()
        train_data_loader = DataLoader(self.train_file_path,self.config.n_positions)
        val_data_loader = DataLoader(self.val_file_path,self.config.n_positions)
        decoder = Decoder(self.vocab_path, special_tokens=["<|endoftext|>","<|im_start|>","<|im_end|>"])
        return train_data_loader, val_data_loader, decoder