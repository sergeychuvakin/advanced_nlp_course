from train_utils import add_paddings
import torch
from config import Config

def val_metrics(model, dl_test, token_id, loss_func):
    
    model.eval()

    # initialize hidden state
    h = model.init_state(Config.BATCH_SIZE)

    for x, y in dl_test:

        if x.shape[0] < Config.BATCH_SIZE:  ## add paddings

            x, y, = add_paddings(x, y, token_id, model)

        # detach hidden states
        h = tuple([each.data for each in h])

        # get the output from the model
        output, h = model(x, h)

        # calculate the loss and perform backprop
        loss = loss_func(output, y.view(-1))
        return loss.item(), torch.exp(loss).item()
        