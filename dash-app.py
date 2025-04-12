from dash import Dash, html, dcc, Input, Output
import plotly.graph_objects as go
import numpy as np
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.datasets import load_breast_cancer

# ============================
# Data and Tree Preparation
# ============================

# Load example dataset
X, y = load_breast_cancer(return_X_y=True)
unique_classes = np.unique(y)
colors = ['blue', 'red']

# Fit multiple trees for demonstration
trees = [DecisionTreeClassifier(max_depth=5, random_state=i).fit(X, y) for i in range(3)]

# Perform global dimensionality reduction
pca = PCA(n_components=2)
X_2d = pca.fit_transform(X)

# Map each tree to a node -> sample index map
def get_node_sample_map(tree, X):
    node_indicator = tree.decision_path(X)
    sample_map = {}
    for node_id in range(tree.tree_.node_count):
        sample_map[node_id] = list(node_indicator[:, node_id].toarray().nonzero()[0])
    return sample_map

node_maps = [get_node_sample_map(t, X) for t in trees]

# ============================
# Dash App Setup
# ============================

app = Dash(__name__)

app.layout = html.Div([
    html.H2("Interactive Decision Tree Visualizer"),
    dcc.Dropdown(id='tree-selector',
                 options=[{'label': f'Tree {i}', 'value': i} for i in range(len(trees))],
                 value=0),
    dcc.Graph(id='tree-graph'),
    dcc.Graph(id='samples-graph')
])

# ============================
# Tree Graph Drawing
# ============================

def draw_tree_structure(tree, highlight_path=None):
    fig = go.Figure()
    positions = {}

    def traverse(node_id, depth=0, x_offset=0.0, width=1.0):
        positions[node_id] = (x_offset + width / 2, -depth)
        left = tree.tree_.children_left[node_id]
        right = tree.tree_.children_right[node_id]
        if left != -1:
            traverse(left, depth+1, x_offset, width/2)
            traverse(right, depth+1, x_offset+width/2, width/2)

    traverse(0)

    for node_id, (x, y) in positions.items():
        fig.add_trace(go.Scatter(
            x=[x], y=[y],
            mode='markers+text',
            marker=dict(size=20),
            text=[str(node_id)],
            customdata=[{
                'id': node_id,
                'feature': tree.tree_.feature[node_id],
                'threshold': tree.tree_.threshold[node_id],
                'impurity': tree.tree_.impurity[node_id],
                'samples': tree.tree_.n_node_samples[node_id],
                'value': tree.tree_.value[node_id].tolist()
            }],
            name=f"Node {node_id}"
        ))

    for node_id, (x, y) in positions.items():
        left = tree.tree_.children_left[node_id]
        right = tree.tree_.children_right[node_id]
        for child in (left, right):
            if child != -1:
                x0, y0 = x, y
                x1, y1 = positions[child]
                fig.add_shape(type="line", x0=x0, y0=y0, x1=x1, y1=y1,
                              line=dict(color="red" if highlight_path and node_id in highlight_path and child in highlight_path else "black", width=3 if highlight_path and node_id in highlight_path and child in highlight_path else 1))

    fig.update_traces(
        hovertemplate=
        "<b>Node %{customdata.id}</b><br>" +
        "Feature: %{customdata.feature}<br>" +
        "Threshold: %{customdata.threshold:.2f}<br>" +
        "Impurity: %{customdata.impurity:.4f}<br>" +
        "Samples: %{customdata.samples}<br>" +
        "Value: %{customdata.value}<extra></extra>"
    )

    fig.update_layout(title="Tree Structure",
                      xaxis=dict(showgrid=False, zeroline=False),
                      yaxis=dict(showgrid=False, zeroline=False),
                      clickmode='event+select')
    return fig

# ============================
# Callbacks
# ============================

@app.callback(
    Output('tree-graph', 'figure'),
    Input('tree-selector', 'value'),
    Input('samples-graph', 'hoverData')
)
def update_tree_figure(tree_id, hoverData):
    highlight_path = None
    if hoverData and 'points' in hoverData:
        point = hoverData['points'][0]
        if 'text' in point and point['text'].startswith('Sample'):
            sample_id = int(point['text'].split()[-1])
            tree = trees[tree_id]
            node_index = tree.decision_path([X[sample_id]]).indices
            highlight_path = set(node_index)
    return draw_tree_structure(trees[tree_id], highlight_path)

@app.callback(
    Output('samples-graph', 'figure'),
    Input('tree-selector', 'value'),
    Input('samples-graph', 'hoverData'),
    Input('tree-graph', 'hoverData')
)
def update_samples_figure(tree_id, hoverData, tree_hover):
    fig = go.Figure()
    highlighted = set()
    tree = trees[tree_id]
    if hoverData and 'points' in hoverData:
        point = hoverData['points'][0]
        if 'text' in point and point['text'].startswith('Sample'):
            sample_id = int(point['text'].split()[-1])
            leaf_id = tree.apply([X[sample_id]])[0]
            highlighted = {i for i in range(len(X)) if tree.apply([X[i]])[0] == leaf_id}
    elif tree_hover and 'points' in tree_hover:
        point = tree_hover['points'][0]
        if 'customdata' in point and isinstance(point['customdata'], dict):
            node_id = point['customdata'].get('id')
            if node_id is not None:
                highlighted = set(node_maps[tree_id].get(node_id, []))

    for cls in unique_classes:
        class_name = 'malignant' if cls == 0 else 'benign'
        class_idx = [i for i in range(len(X)) if y[i] == cls]
        opacity = [1.0 if i in highlighted else 0.1 for i in class_idx]
        fig.add_trace(go.Scatter(
            x=X_2d[class_idx, 0],
            y=X_2d[class_idx, 1],
            mode='markers',
            marker=dict(color=colors[cls], opacity=opacity),
            name=f"Class {class_name}",
            text=[f"Sample {i}" for i in class_idx]
        ))

    x_min, x_max = X_2d[:, 0].min() - 0.5, X_2d[:, 0].max() + 0.5
    y_min, y_max = X_2d[:, 1].min() - 0.5, X_2d[:, 1].max() + 0.5
    fig.update_layout(title="All Samples",
                      xaxis_range=[x_min, x_max],
                      yaxis_range=[y_min, y_max])
    return fig

# ============================
# Run
# ============================

if __name__ == '__main__':
    app.run(debug=True)
