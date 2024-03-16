import gpytorch
import torch 

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel='RBF'):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        if kernel == 'RBF':
            # Assuming you have an RBFKernel instance named `rbf_kernel`:
            rbf_kernel = gpytorch.kernels.RBFKernel()
            # To set an initial length scale:
            rbf_kernel.lengthscale = 0.1 # Evtl. Hyperparameter
            self.covar_module = gpytorch.kernels.ScaleKernel(rbf_kernel)  #45
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)