import torch
import torch.nn as nn


class AudisEconder(nn.Module):
    r"""Args:
        in_dim: the dimension of input tensor
        out_dim: the dimension of output tensor
        H_j: the number of Meta_embeddings
        alpha: the factor of skip-connection
        t: Temperature Coefficient
    """
    def __init__(self, in_dim, out_dim, H_j=20, alpha=0.1, t=1e-5):

        super(AudisEconder, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.w_j = nn.Linear(in_dim, H_j)
        self.leak_relu = nn.LeakyReLU()
        self.W_j = nn.Linear(H_j, H_j)
        self.alpha = alpha
        self.t = t
        self.softmax = nn.Softmax(dim=-1)
        self.ME = nn.Parameter(torch.randn(H_j, out_dim))

    def forward(self, x):
        h_j = self.leak_relu(self.w_j(x))
        x_hat_j = self.W_j(h_j) + self.alpha * h_j
        x_hat_j_h = self.softmax(x_hat_j / self.t)
        e_j = x_hat_j_h @ self.ME
        return e_j



if __name__ == '__main__':
    input = torch.rand(16, 10)
    model = AudisEconder(in_dim=10, out_dim=128)
    out = model(input)
    print(out.shape)
    """
    使用方法：
    self.autodis = AudisEconder(in_dim=len(self.condition_columns), out_dim=len(self.condition_columns))
    """
