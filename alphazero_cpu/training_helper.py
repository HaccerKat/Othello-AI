import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
def loss_fn(prediction, target):
    # could play around with the proportions later
    # policy
    policy_loss = torch.sum(-target[0] * torch.nn.functional.log_softmax(prediction[0], dim=1), dim=1).mean()
    # value
    value_loss = 2 * torch.nn.functional.mse_loss(prediction[1], target[1])
    # print(f"Prediction: {prediction[1]}")
    # print(f"Target: {target[1]}")
    return policy_loss, value_loss

def train_loop(dataloader, model, optimizer, BATCH_SIZE):
    size = len(dataloader.dataset)
    model.train()
    fifths = size // BATCH_SIZE // 5
    for batch, (input, policy, value) in enumerate(dataloader):
        input, policy, value = input.to(device), policy.to(device), value.to(device)
        prediction = model(input)
        policy_loss, value_loss = loss_fn(prediction, (policy, value))
        loss = policy_loss + value_loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        if (batch + 1) % fifths == 0:
            loss, current = loss.item(), batch * BATCH_SIZE
            policy_loss, value_loss = policy_loss.item(), value_loss.item()
            print(f"Loss: {loss:>7f} [{current:>5d}|{size:>5d}]")
            print(f"Policy Loss: {policy_loss:>7f}")
            print(f"Value Loss: {value_loss:>7f}")


def test_loop(dataloader, model):
    model.eval()
    num_batches = len(dataloader)
    policy_loss, value_loss = 0, 0
    with torch.no_grad():
        for input, policy, value in dataloader:
            input, policy, value = input.to(device), policy.to(device), value.to(device)
            prediction = model(input)
            add_policy_loss, add_value_loss = loss_fn(prediction, (policy, value))
            add_policy_loss = add_policy_loss.item()
            add_value_loss = add_value_loss.item()
            policy_loss += add_policy_loss
            value_loss += add_value_loss

    policy_loss /= num_batches
    value_loss /= num_batches
    test_loss = policy_loss + value_loss
    # print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg Loss: {test_loss:>8f} \n")
    print("----------------------------------------------------------------")
    print(f"Avg Loss: {test_loss:>8f}")
    print(f"Policy Loss: {policy_loss:>8f}")
    print(f"Value Loss: {value_loss:>8f} \n")
    return policy_loss, value_loss, test_loss