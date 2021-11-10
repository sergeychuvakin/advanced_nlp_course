import torch
from tqdm import tqdm
from config import Config


def train_model(model,
                train_dl,
                batch_size,
                optimizer,
                loss_func,
                epochs=2,
                print_every=1000,
                clip=1):

    model.train()
    
    torch.autograd.set_detect_anomaly(True)
    
    for e in tqdm(range(epochs)):

        # initialize hidden state
        h = model.init_state(batch_size)
        
        for n, (x, y) in enumerate(tqdm(train_dl)):

            if x.shape[0] < Config.BATCH_SIZE: ## add paddings
              
                x = torch.cat(
                    (
                        x, 
                        torch.full((config.BATCH_SIZE - x.shape[0], 
                                    x.shape[1]),
                                   token_id[Config.TOKEN_PADDING],
                                   device=model.device
                                   )
                        )
                    )
                y = torch.cat(
                    (
                        y, 
                        torch.full((Config.BATCH_SIZE - y.shape[0], 
                                    y.shape[1]),
                                   token_id[Config.TOKEN_PADDING],
                                   device=model.device
                                   )
                        )
                    )
            

            # detach hidden states
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()
            
            # get the output from the model
            output, h = model(x, h)
            
            # calculate the loss and perform backprop
            loss = loss_func(output, y.view(-1))

            # back-propagate error
            loss.backward(retain_graph=True)

            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

            # update weigths
            optimizer.step()            
            
            if n % print_every:
            
                print(f"Cross-entropy loss: {loss}")
              
    return model 