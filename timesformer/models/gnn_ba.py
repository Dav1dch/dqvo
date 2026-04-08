"""
GNN-based Bundle Adjustment model.

A graph neural network that refines camera poses and 3D point positions
by minimizing reprojection error through message passing on a heterogeneous graph.

Architecture:
- Node types: camera (6-DoF pose), point (3D position)
- Edge type: camera observes point (from triangulated tracks)
- Bidirectional message passing between camera and point nodes
"""

import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    """
    Single message passing layer for the BA GNN.

    Message passing direction:
    - Camera -> Point: Camera sends its pose info to observed points
    - Point -> Camera: Points send their 3D position info back to cameras

    Each direction:
    1. Concatenate source embedding + destination embedding + edge features (2D pixel coords)
    2. Pass through MLP to compute message
    3. Aggregate messages by destination node (sum)
    4. Update: concat(node_embedding, aggregated_messages) -> MLP -> residual update
    """

    def __init__(self, hidden_dim):
        super().__init__()
        # Message network: [src_emb, dst_emb, edge_feat] -> hidden_dim
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        # Update network: [node_emb, aggregated_msgs] -> hidden_dim
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, x_dict, edge_index_dict, edge_attr_dict):
        """
        Args:
            x_dict: {'camera': (N_cam, hidden), 'point': (N_pt, hidden)}
            edge_index_dict: {('camera', 'observes', 'point'): (2, N_edges)}
            edge_attr_dict: {('camera', 'observes', 'point'): (N_edges, 2)} normalized u,v
        Returns:
            new_x_dict: updated embeddings for each node type
        """
        new_x_dict = {}
        edge_index = edge_index_dict[("camera", "observes", "point")]
        edge_attr = edge_attr_dict[("camera", "observes", "point")]

        # ---- Point Update (Camera -> Point) ----
        # Each point collects messages from all cameras that observe it
        if "point" in x_dict:
            src = x_dict["camera"][edge_index[0]]  # Source: camera embeddings
            dst = x_dict["point"][edge_index[1]]  # Dest: point embeddings
            msg = self.msg_net(torch.cat([src, dst, edge_attr], dim=-1))
            # Aggregate by point index (sum over all cameras observing this point)
            num_points = x_dict["point"].shape[0]
            agg = torch.zeros(num_points, msg.shape[-1], device=msg.device)
            agg.index_add_(0, edge_index[1], msg)
            point_update = self.update_net(torch.cat([x_dict["point"], agg], dim=-1))
            new_x_dict["point"] = x_dict["point"] + point_update  # Residual update

        # ---- Camera Update (Point -> Camera) ----
        # Each camera collects messages from all points it observes
        if "camera" in x_dict:
            src = x_dict["point"][edge_index[1]]  # Source: point embeddings
            dst = x_dict["camera"][edge_index[0]]  # Dest: camera embeddings
            msg = self.msg_net(torch.cat([src, dst, edge_attr], dim=-1))
            # Aggregate by camera index
            num_cameras = x_dict["camera"].shape[0]
            agg = torch.zeros(num_cameras, msg.shape[-1], device=msg.device)
            agg.index_add_(0, edge_index[0], msg)
            cam_update = self.update_net(torch.cat([x_dict["camera"], agg], dim=-1))
            new_x_dict["camera"] = x_dict["camera"] + cam_update  # Residual update

        return new_x_dict


class GNNBAOptimizer(nn.Module):
    """
    GNN-based Bundle Adjustment optimizer.

    Learns to refine camera poses and 3D point positions by minimizing
    reprojection error through message passing on a heterogeneous graph.

    Architecture:
    - Node types: camera (6-DoF pose), point (3D position)
    - Edge type: camera observes point (from triangulated tracks)
    - Embedding: camera [6], point [3] -> hidden_dim
    - num_layers message passing iterations
    - Output: delta_camera [6], delta_point [3]

    The GNN learns optimized pose corrections (deltas) to be added to input poses.
    Input poses can be either absolute poses or relative poses.
    """

    def __init__(self, hidden_dim=32, num_layers=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Input embeddings: [euler(3), trans(3)] -> hidden for cameras
        #                   [x, y, z] -> hidden for points
        self.camera_embed = nn.Linear(6, hidden_dim)
        self.point_embed = nn.Linear(3, hidden_dim)
        self.edge_embed = nn.Linear(
            2, hidden_dim
        )  # Not used currently (edge feat passed directly)

        # Stack of message passing layers
        self.mp_layers = nn.ModuleList(
            [MessagePassingLayer(hidden_dim) for _ in range(num_layers)]
        )

        # Output projections: hidden -> delta (6 for camera, 3 for point)
        # Camera delta is small correction to be added to input pose
        self.camera_out_proj = nn.Linear(hidden_dim, 6)
        self.point_out_proj = nn.Linear(hidden_dim, 3)

        # Initialize output layers with small weights to ensure small initial deltas
        nn.init.xavier_uniform_(self.camera_out_proj.weight, gain=0.01)
        nn.init.zeros_(self.camera_out_proj.bias)
        nn.init.xavier_uniform_(self.point_out_proj.weight, gain=0.01)
        nn.init.zeros_(self.point_out_proj.bias)

    def forward(self, data, return_delta=False):
        """
        Forward pass through GNN.

        Args:
            data: PyTorch Geometric HeteroData with:
                - data['camera'].x: (N_cam, 6) camera features [euler_norm, trans_norm]
                - data['point'].x: (N_pt, 3) 3D point positions
                - data['camera', 'observes', 'point'].edge_index: (2, N_obs)
                - data['camera', 'observes', 'point'].edge_attr: (N_obs, 2) normalized u,v
            return_delta: if True, return delta to be added to input; if False, return refined pose

        Returns:
            camera: (N_cam, 6) predicted camera parameters (refined pose or delta)
            point: (N_pt, 3) predicted point positions (refined or delta)
        """
        # Embed inputs
        x_dict = {
            "camera": self.camera_embed(data["camera"].x),
            "point": self.point_embed(data["point"].x),
        }
        edge_attr_dict = {
            ("camera", "observes", "point"): data[
                "camera", "observes", "point"
            ].edge_attr
        }
        edge_index_dict = {
            ("camera", "observes", "point"): data[
                "camera", "observes", "point"
            ].edge_index
        }

        # Message passing iterations
        for layer in self.mp_layers:
            x_dict = layer(x_dict, edge_index_dict, edge_attr_dict)

        # Project to output dimension - this is the delta to be added to input
        camera_delta = self.camera_out_proj(x_dict["camera"])
        point_delta = self.point_out_proj(x_dict["point"])

        if return_delta:
            return camera_delta, point_delta
        else:
            # Return refined pose = input + delta
            camera_refined = data["camera"].x + camera_delta
            point_refined = data["point"].x + point_delta
            return camera_refined, point_refined
