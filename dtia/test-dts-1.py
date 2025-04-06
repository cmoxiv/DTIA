

# ## MODEL DETAILS
# t = clf.tree_

# value_squeezed = np.squeeze(t.value)
# value_columns = [f"_value_{i}" for i in range(value_squeezed.T.shape[0])]
# n_samples_for_class_columns =[f"n_samples_for_class_{i}" for i in range(value_squeezed.T.shape[0])]
# internal_columns = ["_weighted_n_node_samples"] + value_columns
# node_columns = "id depth left right feature threshold impurity".split()

# n_samples_for_class_value = t.weighted_n_node_samples[np.newaxis].T * t.value.squeeze()

# df = DataFrame(zip(range(len(t.feature)), t.compute_node_depths(), 
#                    t.children_left, t.children_right,
#                    t.feature, t.threshold, t.impurity, 
#                    *n_samples_for_class_value.T, t.weighted_n_node_samples, *value_squeezed.T),
#                columns=node_columns + n_samples_for_class_columns + internal_columns)


# df['id left right'.split()] = df['id left right'.split()].astype(int)

# df2 = df.iloc[:, :4]

# ## Extracting Paths

# import collections
# from collections import deque

# stack = deque([(int(0), -1, [0])])
# traversed_tree = []

# while len(stack):
#     node, parent, path_so_far = stack.pop()
#     # print(f"{type(node)=} -> {node=}")
#     l, r = df.iloc[node]["left right".split()].astype(int)
#     is_leaf = (l == r)
    
#     print((node, parent, path_so_far))
#     traversed_tree.append((node, parent, path_so_far))
    
#     if not is_leaf:
#         stack.append((r, node, path_so_far + [r]))
#         stack.append((l, node, path_so_far + [l]))
#         pass
#     pass

# print(traversed_tree)
