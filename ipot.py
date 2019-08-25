import torch

def cost(x,y):
    return 1 - (y.mm(x.t()))/(torch.norm(x,2)*torch.norm(y,2))

def ipot(x,y, beta=0.01, K=1):
    n = len(x)
    m = len(y)
    T = torch.ones((n,1)).mm(torch.ones(1,m))
    sigma = (1.0/m)*torch.ones((1,m))
    C = torch.zeros(n,m)
    for i in range(n):
        for j in range(m):
            C[i,j] = cost(x[i],y[j])
    A = torch.zeros(n,m)
    for i in range(n):
        for j in range(m):
            A[i,j] = torch.exp(-C[i,j]*beta)
    for t in range(1):
        Q = A * T
        for k in range(K):
            delta = 1/(n*Q*sigma)
            sigma = 1/(m*Q.t().mm(delta))
        #print(delta.size(),Q.size(), sigma.size())
        T = delta.mm(sigma.mm(Q.t()))
    return T

n,m = 10, 9
x = [torch.randn(1,10) for _ in range(n)]
y = [torch.randn(1,10) for _ in range(m)]
T = ipot(x,y)