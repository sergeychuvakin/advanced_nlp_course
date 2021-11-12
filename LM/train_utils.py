import torch
from config import Config
from processing_utils import load_artifact
from tqdm import tqdm

token_id = load_artifact(Config.SAVE_TOKEN_ID)
id_token = load_artifact(Config.SAVE_ID_TOKEN)


def add_paddings(x, y, token_id, model):

    x = torch.cat(
        (
            x,
            torch.full(
                (Config.BATCH_SIZE - x.shape[0], x.shape[1]),
                token_id[Config.TOKEN_PADDING],
                device=model.device,
            ),
        )
    )

    y = torch.cat(
        (
            y,
            torch.full(
                (Config.BATCH_SIZE - y.shape[0], y.shape[1]),
                token_id[Config.TOKEN_PADDING],
                device=model.device,
            ),
        )
    )
    return x, y


def train_model(
    model,
    train_dl,
    optimizer,
    loss_func,
    epochs=2,
    print_every=1000,
    clip=1,
):

    model.train()

    torch.autograd.set_detect_anomaly(True)

    for e in tqdm(range(epochs)):

        # initialize hidden state
        h = model.init_state(Config.BATCH_SIZE)

        for n, (x, y) in enumerate(tqdm(train_dl)):

            if x.shape[0] < Config.BATCH_SIZE:  ## add paddings

                x, y = add_paddings(x, y, token_id, model)

            # detach hidden states
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            output, h = model(x, h)

            # calculate the loss and perform backprop
            loss = torch.exp(loss_func(output, y.view(-1)))
            # loss = torch.exp(loss) # perplexity
            # back-propagate error
            loss.backward(retain_graph=True)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # update weigths
            optimizer.step()

            if n % print_every:

                print(f"Cross-entropy loss: {loss}")

    return model
