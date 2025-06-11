import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
def loss_fn(prediction, target):
    # could play around with the proportions later
    # policy
    loss = torch.sum(-target[0] * torch.nn.functional.log_softmax(prediction[0], dim=1), dim=1).mean()
    # value
    loss += torch.nn.functional.mse_loss(prediction[1], target[1])
    return loss

def train_loop(dataloader, model, optimizer, BATCH_SIZE):
    size = len(dataloader.dataset)
    model.train()
    fifths = size // BATCH_SIZE // 5
    for batch, (input, policy, value) in enumerate(dataloader):
        input, policy, value = input.to(device), policy.to(device), value.to(device)
        prediction = model(input)
        loss = loss_fn(prediction, (policy, value))
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch + 1) % fifths == 0:
            loss, current = loss.item(), batch * BATCH_SIZE
            print(f"loss: {loss:>7f} [{current:>5d}|{size:>5d}]")


def test_loop(dataloader, model):
    model.eval()
    num_batches = len(dataloader)
    test_loss = 0
    with torch.no_grad():
        for input, policy, value in dataloader:
            input, policy, value = input.to(device), policy.to(device), value.to(device)
            prediction = model(input)
            test_loss += loss_fn(prediction, (policy, value)).item()

    test_loss /= num_batches
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")
    print(f"Avg Loss: {test_loss:>8f} \n")
    return test_loss