import torch
import torchvision
from tqdm import tqdm


class TrainFasterRCNN:

    def __init__(self, model: torchvision, trainDataloader: torch, valDataloader: torch, optimizer: torch,
                 scheduler: torch = None, device: str = 'cuda:0', model_name: str = 'FasterRCNN_v1',
                 iouThreshold: float = 0.5):

        self.model = model
        self.trainDataloader = trainDataloader
        self.valDataloader = valDataloader
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.model_name = model_name
        self.iouThreshold = iouThreshold

        self.model.to(self.device)

    def _train(self):
        print('Training')
        train_loss_list = []

        # initialize tqdm progress bar
        prog_bar = tqdm(self.trainDataloader, total=len(self.trainDataloader))

        for i, data in enumerate(prog_bar):
            self.optimizer.zero_grad()

            images, targets = data

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            train_loss_list.append(loss_value)

            losses.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()

            # update the loss value beside the progress bar for each iteration
            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        train_loss = sum(train_loss_list) / (i + 1)

        return train_loss

    @torch.no_grad()
    def _evolution(self):
        print('Validating')

        val_loss_list = []
        # initialize tqdm progress bar
        prog_bar = tqdm(self.valDataloader, total=len(self.valDataloader))

        for i, data in enumerate(prog_bar):
            images, targets = data

            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            val_loss_list.append(loss_value)

            prog_bar.set_description(desc=f"Loss: {loss_value:.4f}")

        val_loss = sum(val_loss_list) / (i + 1)

        return val_loss

    def train_net(self, num_epochs):

        train_loss_history, val_loss_history = [], []

        for epoch in range(num_epochs):
            print(f"\nEPOCH {epoch + 1} of {num_epochs}")

            train_loss = self._train()
            val_loss = self._evolution()

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            if epoch == 0:
                best_loss = val_loss

            if (val_loss <= best_loss) and (epoch != 0):
                best_loss = val_loss
                try:
                    torch.save(self.model, f'./weights/{self.model_name}_best.pth')
                    print('Save best model.')
                except:
                    print("Can't save best model!")

            try:
                torch.save(self.model, f'./weights/{self.model_name}_last.pth')
            except:
                print("Can't save last model!")

        return {
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history
        }