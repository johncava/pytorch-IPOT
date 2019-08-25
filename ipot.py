import torch

def cost(x,y):
    return 1 - (y.mm(x.t()))/(torch.norm(x,2)*torch.norm(y,2))

def ipot(x,y, beta=0.01, K=1):
    n = len(x)
    m = len(y)
    T = torch.ones((n,1)).mm(torch.ones(m,1).t())
    sigma = (1.0/m)*torch.ones((m,1))
    C = torch.zeros(n,m)
    for i in range(n):
        for j in range(m):
            C[i,j] = cost(x[i],y[j])
    A = torch.zeros(n,m)
    for i in range(n):
        for j in range(m):
            A[i,j] = torch.exp(-C[i,j]*beta)
    for t in range(50):
        Q = A * T
        for k in range(K):
            delta = 1/(n*Q.mm(sigma))
            sigma = 1/(m*Q.t().mm(delta))
        # Squeeze such that [n,1] for delta and [m,1] for sigma turn to [n] and [m]
        # such that torch.diag() can be used to turn delta and sigma into [n,n] and [m,m] matrices
        T = torch.diag(delta.squeeze()).mm(Q).mm(torch.diag(sigma.squeeze()))
    return T

n,m = 10, 9
x = [torch.randn(1,10) for _ in range(n)]
y = [torch.randn(1,10) for _ in range(m)]
T = ipot(x,y)