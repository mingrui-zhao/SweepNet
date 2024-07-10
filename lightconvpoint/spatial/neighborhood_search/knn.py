import torch
from torch_cluster import knn as knn_cluster
from scipy.spatial import KDTree

def knn(points, support_points, K, neighbors_indices=None):

    if neighbors_indices is not None:
        return neighbors_indices

    if K > points.shape[2]:
        K = points.shape[2]
        
    B, D, N = points.shape
    
    flattened = points.transpose(1,2).reshape(B*N, D)

    batch = torch.arange(B).to(points.device)
    batch = torch.repeat_interleave(batch, N)

    pos = flattened
    _, D, S = support_points.shape
    s_pos = support_points.transpose(1,2).reshape(-1, D)
    batch_y = torch.arange(B, device=points.device)
    batch_y = torch.repeat_interleave(batch_y, S)
    
    row, col = knn_cluster(pos, s_pos, K, batch_x=batch, batch_y=batch_y)
    if col.shape[0] == B*S*K:
        result = (col%N).reshape(B, S, K)
        return result
    else:
        pts = points.cpu().detach().transpose(1,2).numpy().copy()
        s_pts = support_points.cpu().detach().transpose(1,2).numpy().copy()
        n = pts.shape[1]
        indices = []
        for i in range(pts.shape[0]):
            tree = KDTree(pts[i])
            _, indices_ = tree.query(s_pts[i], k=K)
            indices.append(torch.tensor(indices_, dtype=torch.long))
        indices = torch.stack(indices, dim=0)
        if K==1:
            indices = indices.unsqueeze(2)
        
    return indices.to(points.device)

# def knn(points, support_points, K, neighbors_indices=None):

#     if neighbors_indices is not None:
#         return neighbors_indices

#     if K > points.shape[2]:
#         K = points.shape[2]
#     pts = points.cpu().detach().transpose(1,2).numpy().copy()
#     s_pts = support_points.cpu().detach().transpose(1,2).numpy().copy()
#     n = pts.shape[1]
#     indices = []
#     for i in range(pts.shape[0]):
#         tree = KDTree(pts[i])
#         _, indices_ = tree.query(s_pts[i], k=K)
#         indices.append(torch.tensor(indices_, dtype=torch.long))
#     indices = torch.stack(indices, dim=0)
#     if K==1:
#         indices = indices.unsqueeze(2)
#     return indices.to(points.device)