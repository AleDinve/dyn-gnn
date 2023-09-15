import numpy as np

def to_categorical(values):
    values = np.array(values)
    classes, indices, count = np.unique(values, return_inverse=True, return_counts=True)
    print(count)
    values_scaled = np.array([i for i in range(classes.size)])
    values =  values_scaled[indices]
    b = np.zeros((values.size, values.max() + 1))
    b[np.arange(values.size), values] = 1
    return list(b)

def wl_equiv_graphs():
    A1 = np.array([[0,1,1,0,0,0],[1,0,0,1,0,0],[1,0,0,1,1,0],
                    [0,1,1,0,0,1],[0,0,1,0,0,1],[0,0,0,1,1,0]])
    A2 = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,1,0,0],
                    [0,0,1,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
    return A1, A2

def cycle_graph(n):

    A = np.zeros((n,n))
    A[0,-1]=1
    A[-1,0]=1
    for i in range(n-1):
        A[i,i+1]=1
        A[i+1,i]=1
    return A
    # s = np.shape(A)[0]
    # G = nx.from_numpy_matrix(A)
    # g = from_networkx(G)
    # g.x = torch.tensor([[1] for i in range(s)], dtype=torch.float)
    # g.y = torch.tensor([[1]],dtype=torch.float)
    # return g

def triangles():
    A = np.array([[0,1,1,0,0,0],[1,0,1,0,0,0],[1,1,0,0,0,0],
                 [0,0,0,0,1,1],[0,0,0,1,0,1],[0,0,0,1,1,0]])
    return A

if __name__== '__main__':
    values = np.random.randint(1,50,50)
    print(to_categorical(values))