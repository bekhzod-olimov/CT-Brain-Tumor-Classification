import os, timm, torch 
from tqdm import tqdm

def train_setup(model_name, epochs, classes, device, lr = 3e-4): 
    m = timm.create_model(model_name, pretrained = True, num_classes = len(classes))  
    return m.to(device), epochs, device, torch.nn.CrossEntropyLoss(), torch.optim.Adam(params = m.parameters(), lr = lr)

def train(tr_dl, val_dl, m, device, loss_fn, optimizer, epochs, save_dir = "saved_models", save_prefix = "med"):
    print("Start training...")
    best_acc = 0
    for epoch in range(epochs):

        epoch_loss, epoch_acc, total = 0, 0, 0
        for idx, batch in tqdm(enumerate(tr_dl)):
            ims, gts = batch
            ims, gts = ims.to(device), gts.to(device)

            total += ims.shape[0]

            preds = m(ims)
            pred_cls = torch.argmax(preds.data, dim = 1)
            loss = loss_fn(preds, gts)

            epoch_acc += (pred_cls == gts).sum().item()
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"{epoch + 1}-epoch train process is completed!")
        print(f"{epoch + 1}-epoch train loss -> {(epoch_loss / len(tr_dl)):.3f}")
        print(f"{epoch + 1}-epoch train accuracy -> {(epoch_acc / total):.3f}")

        m.eval()
        with torch.no_grad():
            val_epoch_loss, val_epoch_acc, val_total = 0, 0, 0
            for idx, batch in enumerate(val_dl):
                ims, gts = batch
                ims, gts = ims.to(device), gts.to(device)
                val_total += ims.shape[0]

                preds = m(ims)
                loss = loss_fn(preds, gts)
                pred_cls = torch.argmax(preds.data, dim = 1)
                val_epoch_acc += (pred_cls == gts).sum().item()
                val_epoch_loss += loss.item()

            val_acc = val_epoch_acc / val_total
            print(f"{epoch + 1}-epoch validation process is completed!")
            print(f"{epoch + 1}-epoch validation loss -> {(val_epoch_loss / len(val_dl)):.3f}")
            print(f"{epoch + 1}-epoch validation accuracy -> {val_acc:.3f}")

            if val_acc > best_acc:
                os.makedirs(save_dir, exist_ok=True)
                best_acc = val_acc
                torch.save(m.state_dict(), f"{save_dir}/{save_prefix}_best_model.pth")
