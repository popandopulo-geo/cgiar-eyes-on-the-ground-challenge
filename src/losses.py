import torch
import torch.nn as nn

class BayesianLoss(nn.Module):
    def __init__(self,
                 classes_prior_proba,
                 costs_proportional,
                 kernel,
                 n_classes):
        
        super(BayesianLoss, self).__init__()

        self.Pi = classes_prior_proba
        self.C = costs_proportional
        self.K = kernel
        self.M = n_classes

        self.cutpoints = nn.Parameter(torch.arange(n_classes - 1).to(torch.float32))

    def forward(self, inputs, targets):
        self.C = self.C.to(inputs.device)
        self.Pi = self.Pi.to(inputs.device)

        # Create a mask for each class
        class_masks = [targets == t for t in range(self.M)]
        class_losses = []

        for t in range(self.M):
            S_t = torch.where(class_masks[t])[0]
            C_t = self.C[:, t]

            # Calculate the difference between cost values
            cost_diff = C_t[:-1] - C_t[1:]

            # Calculate the difference between cutpoints and inputs
            cutpoint_diff = self.cutpoints - inputs[S_t].unsqueeze(1)

            # Calculate the loss for class t
            class_loss = torch.sum(cost_diff * self.K(cutpoint_diff), dim=1) + C_t[-1]

            # Add the class loss weighted by prior probability
            class_losses.append(torch.sum(class_loss) * self.Pi[t])

        return sum(class_losses)