import torch
from loguru import logger


class LM_LSTM(torch.nn.Module):
    """
    LSTM N_GRAM MODEL
    """

    def __init__(
        self,
        vocab_size,
        emb_size,
        logger,
        n_hidden=256,
        n_layers=4,
        drop_prob=0.3,
        lr=0.001,
    ):

        super().__init__()

        self.n_hidden = n_hidden
        self.n_layers = n_layers
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.emb_layer = torch.nn.Embedding(vocab_size, emb_size, device=self.device)

        ## define the LSTM
        self.lstm = torch.nn.LSTM(
            emb_size,
            n_hidden,
            n_layers,
            dropout=drop_prob,
            batch_first=True,
            device=self.device,
        )

        ## define a dropout layer
        self.dropout = torch.nn.Dropout(drop_prob)

        ## define the fully-connected layer
        self.fc = torch.nn.Linear(n_hidden, vocab_size, device=self.device)

    def forward(self, x, hidden):
        """Forward pass through the network.
        These inputs are x, and the hidden/cell state `hidden`."""

        ## pass input through embedding layer
        logger.info(f"x.shape: {x.shape}, {x.device}")
        embedded = self.emb_layer(x)
        logger.info(f"embedded.shape: {embedded.shape}")
        ## Get the outputs and the new hidden state from the lstm
        lstm_output, hidden = self.lstm(embedded, hidden)
        logger.info(f"lstm_output.shape: {lstm_output.shape}")
        ## pass through a dropout layer
        out = self.dropout(lstm_output)
        logger.info(f"dropout out.shape: {out.shape}")
        # out = out.contiguous().view(-1, self.n_hidden)
        out = out.reshape(-1, self.n_hidden)
        logger.info(f"reshape out.shape: {out.shape}")
        ## put "out" through the fully-connected layer
        out = self.fc(out)
        logger.info(f"fc out.shape: {out.shape}")

        # return the final output and the hidden state
        return torch.nn.functional.softmax(out, dim=1), hidden

    ## to pass to first epoch to lstm
    def init_state(self, n_gram_size):
        return (
            torch.zeros(self.n_layers, n_gram_size, self.n_hidden).to(self.device),
            torch.zeros(self.n_layers, n_gram_size, self.n_hidden).to(self.device),
        )


# https://stackoverflow.com/questions/59209086/calculate-perplexity-in-pytorch - perplexity
