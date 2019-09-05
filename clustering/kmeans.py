import torch
import numpy as np
from clustering.pairwise import pairwise_distance

def forgy(X, n_clusters):
	_len = len(X)
	indices = np.random.choice(_len, n_clusters)
	initial_state = X[indices]
	return initial_state


def lloyd1d(X, n_clusters, tol=1e-4, device=None, max_iter=100, init_state=None):
	if device is not None:
		X = X.to(device)

	if init_state is None:
		initial_state = forgy(X, n_clusters).flatten()
	else:
		initial_state = init_state.clone()

	iter = 0
	dis = X.new_empty((n_clusters, X.numel()))
	choice_cluster = X.new_empty(X.numel()).int()
	centers = torch.arange(n_clusters, device=X.device).view(-1, 1).int()
	initial_state_pre = initial_state.clone()
	# temp = X.new_empty((n_clusters, X.numel()))
	while iter < max_iter:
		iter += 1

		# Calculate pair wise distance
		dis[:, ] = X.view(1, -1)
		dis.sub_(initial_state.view(-1, 1))
		dis.pow_(2)

		choice_cluster[:] = torch.argmin(dis, dim=0).int()

		initial_state_pre[:] = initial_state

		temp = X.view(1, -1) * (choice_cluster == centers).float()
		initial_state[:] = temp.sum(1) / (temp != 0).sum(1).float()

		# center_shift = torch.sum(torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2, dim=1)))
		center_shift = torch.sqrt(torch.sum((initial_state - initial_state_pre) ** 2))

		if center_shift < tol:
			break

	return choice_cluster, initial_state
