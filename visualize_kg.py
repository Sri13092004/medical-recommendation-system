"""
Knowledge Graph Visualization Module
Creates interactive visualizations of the healthcare knowledge graph
"""

import networkx as nx
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px
from plotly.offline import plot
import pandas as pd
from typing import Dict, List, Any
import json

class KnowledgeGraphVisualizer:
    """Visualize the healthcare knowledge graph using different methods"""
    
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.node_colors = {
            'disease': '#FF6B6B',
            'symptom': '#4ECDC4', 
            'drug': '#45B7D1',
            'lifestyle': '#96CEB4',
            'nutrition': '#FFEAA7',
            'prevention': '#DDA0DD',
            'severity': '#98D8C8'
        }
    
    def get_node_color(self, node: str) -> str:
        """Get color for node based on its type"""
        node_lower = node.lower()
        if any(disease_word in node_lower for disease_word in ['infection', 'disease', 'syndrome', 'disorder']):
            return self.node_colors['disease']
        elif any(symptom_word in node_lower for symptom_word in ['pain', 'ache', 'fever', 'cough', 'headache']):
            return self.node_colors['symptom']
        elif any(drug_word in node_lower for drug_word in ['medication', 'drug', 'pill', 'tablet']):
            return self.node_colors['drug']
        else:
            return '#CCCCCC'  # Default color
    
    def create_matplotlib_visualization(self, title: str = "Healthcare Knowledge Graph", 
                                     max_nodes: int = 50, figsize: tuple = (15, 10)):
        """Create matplotlib visualization"""
        # Get subgraph with most connected nodes
        degree_centrality = nx.degree_centrality(self.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        
        subgraph = self.graph.subgraph(top_node_names)
        
        plt.figure(figsize=figsize)
        pos = nx.spring_layout(subgraph, k=3, iterations=50)
        
        # Draw nodes
        node_colors = [self.get_node_color(node) for node in subgraph.nodes()]
        nx.draw_networkx_nodes(subgraph, pos, node_color=node_colors, 
                              node_size=300, alpha=0.8)
        
        # Draw edges
        nx.draw_networkx_edges(subgraph, pos, alpha=0.5, edge_color='gray')
        
        # Draw labels
        nx.draw_networkx_labels(subgraph, pos, font_size=8, font_weight='bold')
        
        plt.title(title, fontsize=16, fontweight='bold')
        plt.axis('off')
        plt.tight_layout()
        return plt
    
    def create_plotly_interactive(self, title: str = "Interactive Healthcare Knowledge Graph",
                                max_nodes: int = 30):
        """Create interactive Plotly visualization"""
        # Get subgraph with most connected nodes
        degree_centrality = nx.degree_centrality(self.graph)
        top_nodes = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:max_nodes]
        top_node_names = [node for node, _ in top_nodes]
        
        subgraph = self.graph.subgraph(top_node_names)
        # Use circular layout for better organization
        pos = nx.circular_layout(subgraph, scale=2)
        
        # Prepare data for Plotly
        edge_x = []
        edge_y = []
        for edge in subgraph.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create edge trace with thinner lines
        edge_trace = go.Scatter(x=edge_x, y=edge_y,
                              line=dict(width=0.3, color='#ccc'),
                              hoverinfo='none',
                              mode='lines')
        
        # Prepare node data
        node_x = []
        node_y = []
        node_text = []
        node_colors = []
        node_sizes = []
        
        for node in subgraph.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(f"{node}<br>Connections: {subgraph.degree(node)}")
            node_colors.append(self.get_node_color(node))
            # Smaller, more uniform node sizes
            node_sizes.append(max(15, min(25, subgraph.degree(node) * 1.5)))
        
        # Create node trace with better text positioning
        node_trace = go.Scatter(x=node_x, y=node_y,
                              mode='markers+text',
                              hoverinfo='text',
                              text=[node for node in subgraph.nodes()],
                              textposition="middle center",
                              textfont=dict(size=10, color='black'),
                              hovertext=node_text,
                              marker=dict(size=node_sizes,
                                        color=node_colors,
                                        line=dict(width=1, color='black')))
        
        # Create figure with better layout
        fig = go.Figure(data=[edge_trace, node_trace],
                       layout=go.Layout(title=title,
                                      titlefont_size=18,
                                      showlegend=False,
                                      hovermode='closest',
                                      margin=dict(b=40,l=40,r=40,t=60),
                                      width=1000,
                                      height=800,
                                      annotations=[ dict(
                                          text="Top 30 Most Connected Nodes - Click and drag to explore",
                                          showarrow=False,
                                          xref="paper", yref="paper",
                                          x=0.5, y=-0.1,
                                          xanchor='center', yanchor='top',
                                          font=dict(color='gray', size=14)
                                      )],
                                      xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5]),
                                      yaxis=dict(showgrid=False, zeroline=False, showticklabels=False, range=[-2.5, 2.5])))
        
        return fig
    
    def create_network_statistics(self) -> Dict[str, Any]:
        """Generate network statistics"""
        stats = {
            'total_nodes': self.graph.number_of_nodes(),
            'total_edges': self.graph.number_of_edges(),
            'density': nx.density(self.graph),
            'average_clustering': nx.average_clustering(self.graph),
            'is_connected': nx.is_connected(self.graph),
            'number_of_components': nx.number_connected_components(self.graph)
        }
        
        # Centrality measures
        degree_centrality = nx.degree_centrality(self.graph)
        betweenness_centrality = nx.betweenness_centrality(self.graph)
        
        stats['top_degree_nodes'] = sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        stats['top_betweenness_nodes'] = sorted(betweenness_centrality.items(), key=lambda x: x[1], reverse=True)[:10]
        
        return stats
    
    def save_interactive_html(self, filename: str = "knowledge_graph.html", max_nodes: int = 100):
        """Save interactive visualization as HTML file"""
        fig = self.create_plotly_interactive(max_nodes=max_nodes)
        plot(fig, filename=filename, auto_open=False)
        print(f"✅ Interactive visualization saved as {filename}")
    

def visualize_knowledge_graph():
    """Main function to create visualizations"""
    print("🎨 Creating Knowledge Graph Visualizations...")
    
    # Import the graph from enhanced_main
    try:
        from enhanced_main import G
        if G.number_of_nodes() == 0:
            print("❌ Knowledge graph is empty. Run enhanced_main.py first to build the graph.")
            return
    except ImportError:
        print("❌ Could not import knowledge graph. Make sure enhanced_main.py is in the same directory.")
        return
    
    # Create visualizer
    visualizer = KnowledgeGraphVisualizer(G)
    
    # Generate statistics
    stats = visualizer.create_network_statistics()
    print(f"\n📊 Knowledge Graph Statistics:")
    print(f"   Total Nodes: {stats['total_nodes']}")
    print(f"   Total Edges: {stats['total_edges']}")
    print(f"   Density: {stats['density']:.3f}")
    print(f"   Average Clustering: {stats['average_clustering']:.3f}")
    print(f"   Connected Components: {stats['number_of_components']}")
    
    print(f"\n🔝 Top Connected Nodes:")
    for node, centrality in stats['top_degree_nodes'][:5]:
        print(f"   {node}: {centrality:.3f}")
    
    # Create interactive visualization
    print(f"\n🎨 Creating interactive visualization...")
    visualizer.save_interactive_html("knowledge_graph.html", max_nodes=100)
    
    # Create matplotlib visualization
    print(f"📊 Creating matplotlib visualization...")
    plt = visualizer.create_matplotlib_visualization(max_nodes=50)
    plt.savefig("knowledge_graph.png", dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\n✅ Visualizations created:")
    print(f"   📄 knowledge_graph.html (interactive)")
    print(f"   🖼️  knowledge_graph.png (static)")
    
    return visualizer

if __name__ == "__main__":
    visualize_knowledge_graph()
