import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def masked_loss_fn(pred, tgt, loss_fn, pad_idx=0):
    # Get mask for padding
    mask = tgt.ne(pad_idx)
    # Get number of words in target sentence
    loss_ = loss_fn(pred, tgt)
    loss_ = loss_ * mask.float()
    # Mask out padding
    loss_ = loss_.mean()
    return loss_

def train_loop(model, opt, loss_fn, dataloader, pad_idx):
    
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        img, tgt = batch[0], batch[1]

        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Get mask to mask out the next words
        tgt_mask = model.get_tgt_mask(tgt_input.shape[1]).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(img, tgt_input, tgt_mask)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = masked_loss_fn(pred, tgt_output, loss_fn, pad_idx)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader, pad_idx=0):
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            img, tgt = batch[0], batch[1]

            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Get mask to mask out the next words
            tgt_mask = model.get_tgt_mask(tgt_input.shape[1]).to(device)

            # Standard training except we pass in y_input and tgt_mask
            pred = model(img, tgt_input, tgt_mask)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)      
            loss = masked_loss_fn(pred, tgt_output, loss_fn, pad_idx)
            total_loss += loss.item()
        
    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs, pad_idx = 0):
    
    t_loss, v_loss = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train_loop(model, opt, loss_fn, train_dataloader, pad_idx)
        t_loss += [train_loss]
        validation_loss = validation_loop(model, loss_fn, val_dataloader, pad_idx)
        v_loss += [validation_loss]
        print(f"Training loss: {train_loss:.4f} || Validation loss: {validation_loss:.4f}")
        
    return t_loss, v_loss

def predict(model, img, tgt, max_length=15, SOS_token=2, EOS_token=3, PAD_token=0):

    model.eval()
    y_input = torch.tensor([[SOS_token]], dtype=torch.long)

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(tgt.shape[1])
        
        pred = model(img, tgt, tgt_mask)
        
        # Greedy search
        next_item = pred.topk(1)[1].view(-1)[-1].item() 
        if next_item == PAD_token:
            # If model predicts padding, take the second best option
            next_item = pred.topk(2)[1].view(-1)[-1].item()
        next_item = torch.tensor([[next_item]])

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()