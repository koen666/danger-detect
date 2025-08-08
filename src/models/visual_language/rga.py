import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging


class GraphAttentionLayer(nn.Module):
    """
    Basic graph attention layer for RGA (Relation-aware Graph Attention) networks.
    Based on the paper "Relation-Aware Graph Attention Network for Visual Question Answering"
    """
    
    def __init__(self, in_features, out_features, dropout=0.1, alpha=0.2):
        """
        Initialize the graph attention layer.
        
        Args:
            in_features: Size of each input feature
            out_features: Size of each output feature
            dropout: Dropout probability
            alpha: LeakyReLU negative slope
        """
        super(GraphAttentionLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        
        # Define trainable parameters
        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)
        
        self.a = nn.Parameter(torch.zeros(size=(2*out_features, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        
        # LeakyReLU for attention mechanism
        self.leakyrelu = nn.LeakyReLU(self.alpha)
    
    def forward(self, h, adj):
        """
        Forward pass for the graph attention layer.
        
        Args:
            h: Node features [batch_size, N, in_features]
            adj: Adjacency matrix [batch_size, N, N]
            
        Returns:
            Updated node features [batch_size, N, out_features]
        """
        batch_size, N, _ = h.size()
        
        # Linear transformation
        Wh = torch.matmul(h, self.W)  # [batch_size, N, out_features]
        
        # Prepare for attention
        a_input = torch.cat([Wh.repeat(1, 1, N).view(batch_size, N * N, self.out_features),
                           Wh.repeat(1, N, 1)], dim=2).view(batch_size, N, N, 2 * self.out_features)
        
        # Calculate attention coefficients
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))
        
        # Apply adjacency matrix mask
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        
        # Apply softmax to get attention weights
        attention = F.softmax(attention, dim=2)
        attention = F.dropout(attention, self.dropout, training=self.training)
        
        # Apply attention weights to features
        h_prime = torch.matmul(attention, Wh)
        
        return h_prime


class RGANetwork(nn.Module):
    """
    Relation-aware Graph Attention Network for modeling relationships between entities.
    """
    
    def __init__(self, input_dim, hidden_dim=128, output_dim=64, num_heads=4, dropout=0.1):
        """
        Initialize the RGA network.
        
        Args:
            input_dim: Dimension of input features
            hidden_dim: Dimension of hidden layer
            output_dim: Dimension of output features
            num_heads: Number of attention heads
            dropout: Dropout probability
        """
        super(RGANetwork, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.dropout = dropout
        
        # Multi-head attention layers
        self.attentions = nn.ModuleList([
            GraphAttentionLayer(input_dim, hidden_dim, dropout=dropout, alpha=0.2) 
            for _ in range(num_heads)
        ])
        
        # Second layer attention
        self.out_att = GraphAttentionLayer(
            hidden_dim * num_heads, 
            output_dim, 
            dropout=dropout, 
            alpha=0.2
        )
        
        # Feature embedding layers
        self.fc_embed = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Output projection
        self.fc_out = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(self, x, adj=None):
        """
        Forward pass through the RGA network.
        
        Args:
            x: Input node features [batch_size, num_nodes, input_dim]
            adj: Adjacency matrix [batch_size, num_nodes, num_nodes] or None
            
        Returns:
            Updated node features [batch_size, num_nodes, output_dim]
        """
        batch_size, num_nodes, _ = x.size()
        
        # If adjacency matrix is not provided, create a fully connected graph
        if adj is None:
            adj = torch.ones(batch_size, num_nodes, num_nodes).to(x.device)
        
        # Initial feature embedding
        x_embed = self.fc_embed(x)
        
        # Apply graph attention layers and concatenate results from each head
        x = torch.cat([att(x, adj) for att in self.attentions], dim=2)
        
        # Apply final attention layer
        x = F.elu(self.out_att(x, adj))
        
        # Apply output projection
        x = self.fc_out(x)
        
        return x


class RelationGraphBuilder:
    """
    Utility class for building relation graphs from multimodal features.
    """
    
    def __init__(self, feature_dim=1024, hidden_dim=256, output_dim=128, num_heads=4, threshold=0.5):
        """
        Initialize the relation graph builder.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dim: Hidden dimension for the RGA network
            output_dim: Output dimension for the RGA network
            num_heads: Number of attention heads
            threshold: Similarity threshold for building adjacency matrix
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('RelationGraphBuilder')
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_heads = num_heads
        self.threshold = threshold
        
        # Initialize the RGA network
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.rga_network = RGANetwork(
            input_dim=feature_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            num_heads=num_heads
        ).to(self.device)
        
        self.is_initialized = True
    
    def build_adjacency_matrix(self, features, threshold=None):
        """
        Build an adjacency matrix from node features based on similarity.
        
        Args:
            features: Node features [num_nodes, feature_dim]
            threshold: Similarity threshold (optional, defaults to self.threshold)
            
        Returns:
            Adjacency matrix [num_nodes, num_nodes]
        """
        if threshold is None:
            threshold = self.threshold
        
        # Compute pairwise cosine similarity
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.transpose(0, 1))
        
        # Apply threshold to create binary adjacency matrix
        adj = (similarity > threshold).float()
        
        # Ensure self-connections
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        return adj
    
    def process_features(self, features, adj=None):
        """
        Process features through the RGA network.
        
        Args:
            features: Node features [batch_size, num_nodes, feature_dim] or [num_nodes, feature_dim]
            adj: Adjacency matrix (optional)
            
        Returns:
            Updated node features with relational context
        """
        if not self.is_initialized:
            self.logger.error("RGA network not initialized")
            return None
        
        # Handle non-batched input
        if len(features.shape) == 2:
            features = features.unsqueeze(0)  # [1, num_nodes, feature_dim]
        
        # Convert numpy arrays to torch tensors if necessary
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Move to device
        features = features.to(self.device)
        
        # Build adjacency matrix if not provided
        if adj is None:
            adj = torch.zeros(features.size(0), features.size(1), features.size(1))
            for i in range(features.size(0)):
                adj[i] = self.build_adjacency_matrix(features[i])
            adj = adj.to(self.device)
        elif isinstance(adj, np.ndarray):
            adj = torch.from_numpy(adj).float().to(self.device)
        
        # Process features through RGA network
        self.rga_network.eval()
        with torch.no_grad():
            processed_features = self.rga_network(features, adj)
        
        # Convert back to numpy if input was numpy
        if isinstance(features, np.ndarray):
            processed_features = processed_features.cpu().numpy()
        
        return processed_features
    
    def extract_relations(self, features, threshold=None):
        """
        Extract relations between entities based on feature similarity.
        
        Args:
            features: Entity features [num_entities, feature_dim]
            threshold: Similarity threshold (optional)
            
        Returns:
            Dictionary containing adjacency matrix and relation strengths
        """
        if not self.is_initialized:
            self.logger.error("RGA network not initialized")
            return None
        
        # Convert numpy arrays to torch tensors if necessary
        if isinstance(features, np.ndarray):
            features = torch.from_numpy(features).float()
        
        # Move to device
        features = features.to(self.device)
        
        # Compute pairwise similarity
        features_norm = F.normalize(features, p=2, dim=1)
        similarity = torch.mm(features_norm, features_norm.transpose(0, 1))
        
        # Build adjacency matrix
        if threshold is None:
            threshold = self.threshold
        adj = (similarity > threshold).float()
        
        # Ensure self-connections
        adj = adj + torch.eye(adj.size(0), device=adj.device)
        
        # Process features through RGA network
        processed_features = self.process_features(features.unsqueeze(0), adj.unsqueeze(0))
        
        # Extract top relations
        num_entities = features.size(0)
        relation_strengths = similarity.cpu().numpy()
        
        # Convert to numpy for return
        adj_np = adj.cpu().numpy()
        processed_features_np = processed_features.squeeze(0).cpu().numpy()
        
        return {
            'adjacency_matrix': adj_np,
            'relation_strengths': relation_strengths,
            'processed_features': processed_features_np
        }


class SceneGraphBuilder:
    """
    Builds a scene graph from visual and textual features using the RGA module.
    """
    
    def __init__(self, relation_model=None, num_detections=10):
        """
        Initialize the scene graph builder.
        
        Args:
            relation_model: RGA model for relation processing
            num_detections: Maximum number of objects to detect
        """
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('SceneGraphBuilder')
        
        self.num_detections = num_detections
        
        # Use provided relation model or create a new one
        if relation_model is None:
            self.relation_model = RelationGraphBuilder()
        else:
            self.relation_model = relation_model
            
        # Initialize with default device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.is_initialized = self.relation_model.is_initialized
    
    def build_scene_graph_from_caption(self, frame, caption, qwen_extractor=None):
        """
        Build a simple scene graph from a video frame and its caption.
        
        Args:
            frame: Video frame
            caption: Caption describing the frame
            qwen_extractor: QwenVLFeatureExtractor instance (optional)
            
        Returns:
            Scene graph dictionary
        """
        import re
        
        # Extract features if QwenVL extractor is provided
        if qwen_extractor is not None:
            features = qwen_extractor.extract_features(frame)
            if features is None:
                self.logger.error("Failed to extract features")
                return None
        else:
            # Use placeholder features if extractor not provided
            features = np.random.randn(1, 1024).astype(np.float32)
        
        # Parse caption to extract entities and relations
        # This is a simple rule-based approach, could be replaced with NLP tools
        entities = []
        relations = []
        
        # Extract nouns as entities
        nouns = re.findall(r'\b[A-Za-z]+\b', caption)
        nouns = [n.lower() for n in nouns if len(n) > 3]  # Simple filtering
        
        # Deduplicate
        entities = list(set(nouns))[:self.num_detections]
        
        # Create placeholder features for entities
        num_entities = len(entities)
        if num_entities < 2:
            self.logger.warning("Not enough entities found in caption")
            return {
                'nodes': entities,
                'edges': [],
                'caption': caption
            }
        
        # Create entity features (placeholder or from model)
        entity_features = np.random.randn(num_entities, self.relation_model.feature_dim).astype(np.float32)
        
        # Extract relations using RGA
        relations_data = self.relation_model.extract_relations(
            torch.from_numpy(entity_features).float()
        )
        
        if relations_data is None:
            self.logger.error("Failed to extract relations")
            return None
        
        # Build edges from adjacency matrix
        edges = []
        adj_matrix = relations_data['adjacency_matrix']
        relation_strengths = relations_data['relation_strengths']
        
        for i in range(num_entities):
            for j in range(num_entities):
                if i != j and adj_matrix[i, j] > 0:
                    edges.append({
                        'from': entities[i],
                        'to': entities[j],
                        'weight': float(relation_strengths[i, j])
                    })
        
        # Sort edges by weight
        edges.sort(key=lambda x: x['weight'], reverse=True)
        
        return {
            'nodes': entities,
            'edges': edges,
            'caption': caption
        }
    
    def visualize_scene_graph(self, graph, output_path=None):
        """
        Visualize the scene graph.
        
        Args:
            graph: Scene graph dictionary
            output_path: Path to save the visualization (optional)
            
        Returns:
            Visualization image
        """
        if graph is None:
            self.logger.error("No graph to visualize")
            return None
        
        try:
            import matplotlib.pyplot as plt
            import networkx as nx
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
            import cv2
            
            # Create a networkx graph
            G = nx.DiGraph()
            
            # Add nodes
            for i, node in enumerate(graph['nodes']):
                G.add_node(node)
            
            # Add edges
            for edge in graph['edges']:
                G.add_edge(edge['from'], edge['to'], weight=edge['weight'])
            
            # Create figure and axis
            fig = Figure(figsize=(10, 8))
            canvas = FigureCanvas(fig)
            ax = fig.add_subplot(111)
            
            # Position nodes using spring layout
            pos = nx.spring_layout(G)
            
            # Draw nodes
            nx.draw_networkx_nodes(G, pos, node_size=700, node_color='lightblue', ax=ax)
            
            # Draw node labels
            nx.draw_networkx_labels(G, pos, font_size=12, ax=ax)
            
            # Draw edges with weights as width
            edge_weights = [G[u][v]['weight'] * 2 for u, v in G.edges()]
            nx.draw_networkx_edges(G, pos, width=edge_weights, alpha=0.7, 
                                 edge_color='gray', arrows=True, ax=ax)
            
            # Add caption as title
            ax.set_title(graph['caption'], fontsize=14)
            
            # Remove axis
            ax.axis('off')
            
            # Render the figure to a numpy array
            canvas.draw()
            img = np.array(canvas.renderer.buffer_rgba())
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            # Save if output path is provided
            if output_path:
                cv2.imwrite(output_path, img)
            
            return img
            
        except ImportError as e:
            self.logger.error(f"Failed to import visualization libraries: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Error visualizing graph: {str(e)}")
            return None

    def visualize_graph(self, graph, output_path=None):
        """
        Alias for visualize_scene_graph method to ensure interface compatibility.
        
        Args:
            graph: Scene graph dictionary
            output_path: Path to save the visualization (optional)
            
        Returns:
            Visualization image
        """
        return self.visualize_scene_graph(graph, output_path)


# Simple test
if __name__ == "__main__":
    # Create a relation graph builder
    builder = RelationGraphBuilder(feature_dim=512, hidden_dim=128, output_dim=64)
    
    # Create random features for testing
    num_entities = 5
    features = torch.randn(1, num_entities, 512)
    
    # Build adjacency matrix
    adj = builder.build_adjacency_matrix(features[0])
    print(f"Adjacency matrix shape: {adj.shape}")
    
    # Process features
    processed = builder.process_features(features, adj.unsqueeze(0))
    print(f"Processed features shape: {processed.shape}")
    
    # Test scene graph builder
    scene_builder = SceneGraphBuilder(builder)
    
    # Create a test caption
    caption = "A person is sitting on a chair next to a table with a computer."
    
    # Create a dummy frame
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Build scene graph
    graph = scene_builder.build_scene_graph_from_caption(frame, caption)
    
    if graph:
        print(f"Scene graph nodes: {graph['nodes']}")
        print(f"Scene graph edges: {graph['edges']}")
        
        # Visualize graph
        img = scene_builder.visualize_scene_graph(graph, "scene_graph_test.jpg")
        if img is not None:
            print("Scene graph visualization saved to scene_graph_test.jpg") 