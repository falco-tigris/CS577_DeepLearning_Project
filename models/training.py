import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train_loop(model, opt, loss_fn, dataloader):
    
    model.train()
    total_loss = 0
    
    for batch in dataloader:
        img, tgt = batch[0], batch[1]
        
        # Get mask to mask out the next words
        # TODO: ADD MASK FOR PADDING
        # tgt_mask = model.get_tgt_mask(len(tgt)).to(device)

        # Standard training except we pass in y_input and tgt_mask
        pred = model(img, tgt)

        # Permute pred to have batch size first again
        pred = pred.permute(1, 2, 0)      
        loss = loss_fn(pred, tgt)

        opt.zero_grad()
        loss.backward()
        opt.step()
    
        total_loss += loss.item()
        
    return total_loss / len(dataloader)


def validation_loop(model, loss_fn, dataloader):
    
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in dataloader:
            img, tgt = batch[0], batch[1]
            
            # Get mask to mask out the next words
            # tgt_mask = model.get_tgt_mask(len(tgt)).to(device)

            # Standard training except we pass in y_input and tgt_mask
            pred = model(img, tgt)

            # Permute pred to have batch size first again
            pred = pred.permute(1, 2, 0)      
            loss = loss_fn(pred, tgt)
            total_loss += loss.item()
        
    return total_loss / len(dataloader)


def fit(model, opt, loss_fn, train_dataloader, val_dataloader, epochs):
    
    train_loss_list, validation_loss_list = [], []
    for epoch in range(epochs):
        print(f"Epoch {epoch + 1}")
        train_loss = train_loop(model, opt, loss_fn, train_dataloader)
        train_loss_list += [train_loss]
        validation_loss = validation_loop(model, loss_fn, val_dataloader)
        validation_loss_list += [validation_loss]
        print(f"Training loss: {train_loss:.4f} || Validation loss: {validation_loss:.4f}")
        
    return train_loss_list, validation_loss_list

def predict(model, img, tgt, max_length=15, SOS_token=2, EOS_token=3):

    model.eval()
    y_input = torch.tensor([[SOS_token]], dtype=torch.long)

    for _ in range(max_length):
        # Get source mask
        tgt_mask = model.get_tgt_mask(y_input.size(1))
        
        pred = model(img, tgt)
        
        # Greedy search
        next_item = pred.topk(1)[1].view(-1)[-1].item() 
        next_item = torch.tensor([[next_item]])

        # Concatenate previous input with predicted best word
        y_input = torch.cat((y_input, next_item), dim=1)

        # Stop if model predicts end of sentence
        if next_item.view(-1).item() == EOS_token:
            break

    return y_input.view(-1).tolist()