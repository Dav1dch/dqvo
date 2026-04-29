"""
GNN-based Bundle Adjustment model.

Refactored architecture with:
- Node types: camera, point
- Edge types: point->camera observations, camera->camera temporal links
- Update order per layer: c2c edge → point->camera → camera->point → c2c edge
- Deeper layers with LayerNorm + residual connections

The final optimized pose output is read from camera-to-camera edge features.
"""

import numpy as np
import torch
import torch.nn as nn


class PointToCameraLayer(nn.Module):
    """
    Update camera node embeddings using point nodes and point-to-camera edges.

    Message direction is strictly: point -> camera.
    Uses LayerNorm + residual connection for deeper architectures.
    """

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, camera_hidden, point_hidden, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return camera_hidden

        src_point = point_hidden[edge_index[0]]
        dst_camera = camera_hidden[edge_index[1]]
        msg = self.msg_net(torch.cat([src_point, dst_camera, edge_attr], dim=-1))

        agg = torch.zeros(
            camera_hidden.shape[0], msg.shape[-1], device=msg.device, dtype=msg.dtype
        )
        agg.index_add_(0, edge_index[1], msg)
        cam_update = self.update_net(torch.cat([camera_hidden, agg], dim=-1))
        return self.norm(camera_hidden + cam_update)


class CameraToPointLayer(nn.Module):
    """
    Update point node embeddings using camera nodes and camera-to-point edges.

    Message direction is strictly: camera -> point.
    Enables iterative point refinement informed by camera pose updates.
    Uses LayerNorm + residual connection for deeper architectures.
    """

    def __init__(self, hidden_dim, edge_dim):
        super().__init__()
        self.msg_net = nn.Sequential(
            nn.Linear(hidden_dim * 2 + edge_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.update_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, point_hidden, camera_hidden, edge_index, edge_attr):
        if edge_index.numel() == 0:
            return point_hidden

        src_camera = camera_hidden[edge_index[0]]
        dst_point = point_hidden[edge_index[1]]
        msg = self.msg_net(torch.cat([src_camera, dst_point, edge_attr], dim=-1))

        agg = torch.zeros(
            point_hidden.shape[0], msg.shape[-1], device=msg.device, dtype=msg.dtype
        )
        agg.index_add_(0, edge_index[1], msg)
        pt_update = self.update_net(torch.cat([point_hidden, agg], dim=-1))
        return self.norm(point_hidden + pt_update)


class CameraEdgeUpdateLayer(nn.Module):
    """
    Update camera-to-camera edge embeddings using updated camera nodes.

    Edge direction is single-directional and should follow frame index order.
    Uses LayerNorm + residual connection for deeper architectures.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.edge_update = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)

    def forward(self, camera_hidden, edge_index, edge_hidden):
        if edge_index.numel() == 0:
            return edge_hidden

        cam_src = camera_hidden[edge_index[0]]
        cam_dst = camera_hidden[edge_index[1]]
        edge_delta = self.edge_update(torch.cat([cam_src, cam_dst, edge_hidden], dim=-1))
        return self.norm(edge_hidden + edge_delta)


class GNNBAOptimizer(nn.Module):
    """
    GNN-based Bundle Adjustment optimizer with two node/edge types.

    Node types:
    - camera: optional feature (default 6D pose feature or frame-index feature)
    - point: 3D coordinates

    Edge types:
    - point -> camera: observation edge with image-coordinate feature
    - camera -> camera: temporal edge with relative pose feature

    Update order per layer:
    1) update c2c edge embeddings from current camera nodes
    2) update camera nodes from points + point-to-camera edges
    3) update point nodes from cameras + camera-to-point edges
    4) update c2c edge embeddings from updated camera nodes

    The optimized relative pose is read out from updated camera-to-camera edges.
    """

    def __init__(
        self,
        hidden_dim=32,
        num_layers=6,
        camera_feat_dim=6,
        point_feat_dim=3,
        p2c_edge_dim=2,
        c2c_edge_dim=6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.camera_feat_dim = camera_feat_dim
        self.point_feat_dim = point_feat_dim
        self.p2c_edge_dim = p2c_edge_dim
        self.c2c_edge_dim = c2c_edge_dim

        self.camera_embed = nn.Linear(camera_feat_dim, hidden_dim)
        self.point_embed = nn.Linear(point_feat_dim, hidden_dim)
        self.p2c_edge_embed = nn.Linear(p2c_edge_dim, hidden_dim)
        self.c2c_edge_embed = nn.Linear(c2c_edge_dim, hidden_dim)

        self.point_to_camera_layers = nn.ModuleList(
            [PointToCameraLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.camera_to_point_layers = nn.ModuleList(
            [CameraToPointLayer(hidden_dim, hidden_dim) for _ in range(num_layers)]
        )
        self.camera_edge_layers = nn.ModuleList(
            [CameraEdgeUpdateLayer(hidden_dim) for _ in range(num_layers)]
        )

        # Compatibility heads
        self.camera_out_proj = nn.Linear(hidden_dim, 6)
        self.point_out_proj = nn.Linear(hidden_dim, 3)
        self.c2c_pose_out_proj = nn.Linear(hidden_dim, 6)

        # Small init keeps initial residual corrections stable.
        nn.init.xavier_uniform_(self.camera_out_proj.weight, gain=0.01)
        nn.init.zeros_(self.camera_out_proj.bias)
        nn.init.xavier_uniform_(self.point_out_proj.weight, gain=1.0)
        nn.init.zeros_(self.point_out_proj.bias)
        nn.init.xavier_uniform_(self.c2c_pose_out_proj.weight, gain=0.01)
        nn.init.zeros_(self.c2c_pose_out_proj.bias)

    def _resolve_p2c_edges(self, data):
        """
        Resolve point->camera edges.

        Supports explicit ('point', *, 'camera') edges; falls back to reversing
        legacy ('camera', 'observes', 'point') edges.
        """
        for edge_type in data.edge_types:
            if edge_type[0] == "point" and edge_type[2] == "camera":
                edge_store = data[edge_type]
                edge_index = edge_store.edge_index
                edge_attr = edge_store.edge_attr
                return edge_index, edge_attr

        legacy_type = ("camera", "observes", "point")
        if legacy_type in data.edge_types:
            edge_store = data[legacy_type]
            edge_index = edge_store.edge_index
            edge_attr = edge_store.edge_attr
            reversed_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
            return reversed_index, edge_attr

        raise ValueError(
            "No point-to-camera edges found. Expect ('point', *, 'camera') "
            "or legacy ('camera', 'observes', 'point')."
        )

    def _resolve_c2p_edges(self, data):
        """
        Resolve camera->point edges for point refinement.

        Searches for ('camera', *, 'point') edges (original direction).
        Falls back to reversing ('point', *, 'camera') edges.
        """
        for edge_type in data.edge_types:
            if edge_type[0] == "camera" and edge_type[2] == "point":
                edge_store = data[edge_type]
                edge_index = edge_store.edge_index
                edge_attr = edge_store.edge_attr
                return edge_index, edge_attr

        for edge_type in data.edge_types:
            if edge_type[0] == "point" and edge_type[2] == "camera":
                edge_store = data[edge_type]
                edge_index = edge_store.edge_index
                edge_attr = edge_store.edge_attr
                reversed_index = torch.stack([edge_index[1], edge_index[0]], dim=0)
                return reversed_index, edge_attr

        raise ValueError(
            "No camera-to-point edges found. Expect ('camera', *, 'point') "
            "or ('point', *, 'camera')."
        )

    def _resolve_c2c_edges(self, data):
        """
        Resolve camera->camera temporal edges and relative-pose attributes.

        If c2c edges are missing, build default chain edges i->i+1 and initialize
        edge_attr from camera feature differences when available.
        """
        for edge_type in data.edge_types:
            if edge_type[0] == "camera" and edge_type[2] == "camera":
                edge_store = data[edge_type]
                edge_index = edge_store.edge_index
                mask = edge_index[0] < edge_index[1]
                if mask.any():
                    edge_index = edge_index[:, mask]
                    if hasattr(edge_store, "edge_attr") and edge_store.edge_attr is not None:
                        edge_attr = edge_store.edge_attr[mask]
                    else:
                        edge_attr = None
                else:
                    edge_attr = edge_store.edge_attr if hasattr(edge_store, "edge_attr") else None
                break
        else:
            num_cam = data["camera"].x.shape[0]
            if num_cam < 2:
                edge_index = torch.zeros(2, 0, dtype=torch.long, device=data["camera"].x.device)
                edge_attr = torch.zeros(
                    0,
                    self.c2c_edge_dim,
                    dtype=data["camera"].x.dtype,
                    device=data["camera"].x.device,
                )
                return edge_index, edge_attr

            src = torch.arange(0, num_cam - 1, device=data["camera"].x.device)
            dst = src + 1
            edge_index = torch.stack([src, dst], dim=0)
            edge_attr = None

        if edge_attr is None:
            cam_x = data["camera"].x
            if cam_x.shape[-1] >= self.c2c_edge_dim and edge_index.shape[1] > 0:
                diff = cam_x[edge_index[1], : self.c2c_edge_dim] - cam_x[
                    edge_index[0], : self.c2c_edge_dim
                ]
                # Normalize so GNN output matches the normalized loss space
                std_a = np.array([2.8256e-3, 1.7771e-2, 3.2326e-3])
                std_t = np.array([2.5584e-2, 1.8545e-2, 3.0352e-1])
                std_vec = np.concatenate([std_a, std_t])
                std_tensor = torch.tensor(std_vec, dtype=diff.dtype, device=diff.device)
                edge_attr = diff / std_tensor
            else:
                edge_attr = torch.zeros(
                    edge_index.shape[1],
                    self.c2c_edge_dim,
                    dtype=cam_x.dtype,
                    device=cam_x.device,
                )

        return edge_index, edge_attr

    def forward(self, data, return_delta=False, output_mode="absolute"):
        """
        Forward pass through GNN with per-layer update order:
        c2c edge → point->camera → camera->point → c2c edge.

        Args:
            data: PyTorch Geometric HeteroData with:
                - data['camera'].x: (N_cam, C_cam) camera features (pose or frame-index)
                - data['point'].x: (N_pt, 3) point coordinates
                - point->camera edge index/attr (or legacy camera->point, auto-reversed)
                - camera->camera edge index/attr (or auto-built chain i->i+1)
            return_delta: if True, return legacy camera/point deltas.
            output_mode: one of ["absolute", "relative"].
                - "absolute": return refined camera node outputs for compatibility
                - "relative": return optimized camera-to-camera edge poses directly (recommended)

        Returns:
            if return_delta=True:
                camera_delta: (N_cam, 6)
                point_delta: (N_pt, 3)
            else:
                if output_mode="absolute":
                    camera_refined: (N_cam, 6)
                    point_refined: (N_pt, 3)
                if output_mode="relative":
                    camera_relative: (N_edge_c2c, 6)  # optimized relative poses
                    point_refined: (N_pt, 3)
        """
        cam_x = data["camera"].x
        pt_x = data["point"].x

        p2c_edge_index, p2c_edge_attr = self._resolve_p2c_edges(data)
        c2p_edge_index, c2p_edge_attr = self._resolve_c2p_edges(data)
        c2c_edge_index, c2c_edge_attr = self._resolve_c2c_edges(data)

        camera_hidden = self.camera_embed(cam_x)
        point_hidden = self.point_embed(pt_x)
        p2c_edge_hidden = self.p2c_edge_embed(p2c_edge_attr)
        c2p_edge_hidden = self.p2c_edge_embed(c2p_edge_attr)
        c2c_edge_hidden = self.c2c_edge_embed(c2c_edge_attr)

        # Per-layer updates: c2c edge → point->camera → camera->point → c2c edge
        for pt_cam_layer, cam_to_pt_layer, cam_edge_layer in zip(
            self.point_to_camera_layers,
            self.camera_to_point_layers,
            self.camera_edge_layers,
        ):
            c2c_edge_hidden = cam_edge_layer(
                camera_hidden, c2c_edge_index, c2c_edge_hidden
            )
            camera_hidden = pt_cam_layer(
                camera_hidden, point_hidden, p2c_edge_index, p2c_edge_hidden
            )
            point_hidden = cam_to_pt_layer(
                point_hidden, camera_hidden, c2p_edge_index, c2p_edge_hidden
            )
            c2c_edge_hidden = cam_edge_layer(
                camera_hidden, c2c_edge_index, c2c_edge_hidden
            )

        # Compatibility outputs
        camera_delta = self.camera_out_proj(camera_hidden)
        point_delta = self.point_out_proj(point_hidden)

        if return_delta:
            return camera_delta, point_delta

        point_refined = pt_x + point_delta

        if output_mode == "absolute":
            if cam_x.shape[-1] == 6:
                camera_refined = cam_x + camera_delta
            else:
                camera_refined = camera_delta
            return camera_refined, point_refined

        if output_mode == "relative":
            c2c_delta = self.c2c_pose_out_proj(c2c_edge_hidden)
            camera_relative = c2c_delta  # GNN directly outputs optimized relative pose
            return camera_relative, point_refined

        raise ValueError(f"Unsupported output_mode: {output_mode}")
