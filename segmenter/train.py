import sys
sys.path.insert(0, 'segmenter')

from dataset import PeopleDataset, get_transform
from model import get_model_instance_segmentation
from engine import train_one_epoch
import utils
import torch

def main(num_epochs=10, num_classes=2):

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    dataset = PeopleDataset('Human-Segmentation-Dataset', get_transform(train=True))
    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=4,
        collate_fn=utils.collate_fn)

    model = get_model_instance_segmentation(num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005,
                                momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                   step_size=3,
                                                   gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()

    return model.to('cpu').eval()

if __name__ == '__main__':

    model = main()
    torch.save(model.state_dict(), 'model.bin')