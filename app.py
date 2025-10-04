import streamlit as st
import networkx as nx
import numpy as np
from pyvis.network import Network
from typing import Tuple


st.set_page_config(page_title="Network Simulator", layout="wide")


def generate_graph(model: str, n: int, m: int, regularity: float, seed: int) -> nx.Graph:
    if n < 2:
        return nx.empty_graph(n)

    # Bound m to valid range
    m = max(1, min(m, n - 1))

    if model == "Watts–Strogatz (regularity)":
        # k must be even in WS; map regularity r∈[0,1] to rewiring p = 1 - r
        k = m if m % 2 == 0 else m - 1
        k = max(2, k)
        p = float(max(0.0, min(1.0, 1.0 - regularity)))
        return nx.watts_strogatz_graph(n=n, k=k, p=p, seed=seed)

    if model == "Random regular (degree = m)":
        # For a random regular graph, n*m must be even and m < n
        if (n * m) % 2 == 1:
            m = m + 1 if m + 1 < n else m - 1
        m = max(1, min(m, n - 1))
        return nx.random_regular_graph(d=m, n=n, seed=seed)

    if model == "Erdős–Rényi (avg degree ≈ m)":
        # Approximate average degree by setting edge probability p ≈ m/(n-1)
        p = float(m) / float(max(1, n - 1))
        p = float(max(0.0, min(1.0, p)))
        return nx.erdos_renyi_graph(n=n, p=p, seed=seed)

    if model == "Barabási–Albert (attach m edges)":
        m_ba = max(1, min(m, n - 1))
        return nx.barabasi_albert_graph(n=n, m=m_ba, seed=seed)

    # Fallback: simple ring lattice with m nearest neighbors (even m)
    k = m if m % 2 == 0 else m - 1
    k = max(2, k)
    return nx.watts_strogatz_graph(n=n, k=k, p=0.0, seed=seed)


def graph_metrics(G: nx.Graph) -> Tuple[int, int, float, float, int, float | None]:
    n = G.number_of_nodes()
    e = G.number_of_edges()
    degs = [d for _, d in G.degree()]
    avg_deg = float(np.mean(degs)) if degs else 0.0
    clustering = nx.average_clustering(G) if n > 1 else 0.0
    components = nx.number_connected_components(G)

    apl: float | None = None
    if components == 1 and n <= 2000 and e > 0:
        try:
            apl = nx.average_shortest_path_length(G)
        except Exception:
            apl = None
    return n, e, avg_deg, clustering, components, apl


def render_pyvis(G: nx.Graph, height_px: int, physics: bool, dark: bool) -> str:
    bg = "#0f1116" if dark else "#ffffff"
    font = "#e0e0e0" if dark else "#2b2b2b"
    net = Network(height=f"{height_px}px", width="100%", notebook=False, bgcolor=bg, font_color=font)

    # Improve layout stability for medium/large graphs
    if physics:
        net.barnes_hut(gravity=-5000, central_gravity=0.2, spring_length=120, spring_strength=0.05, damping=0.9)
    else:
        net.set_options("""
{
  "physics": { "enabled": false },
  "interaction": { "hover": true }
}
        """)

    # Map nodes with degree-based color
    degrees = dict(G.degree())
    max_deg = max(degrees.values()) if degrees else 1

    def color_for_degree(d):
        # blue( low ) -> green -> yellow -> red( high )
        t = d / max_deg if max_deg > 0 else 0
        r = int(255 * max(0, min(1, 2 * (t - 0.5))))
        g = int(255 * (1 - abs(2 * t - 1)))
        b = int(255 * max(0, min(1, 1 - 2 * t)))
        return f"#{r:02x}{g:02x}{b:02x}"

    for node in G.nodes():
        d = degrees.get(node, 0)
        net.add_node(
            node,
            label=str(node),
            title=f"Node {node} | degree {d}",
            color=color_for_degree(d),
        )

    for u, v in G.edges():
        net.add_edge(u, v)

    # Return HTML string to embed in Streamlit
    return net.generate_html()  # pyvis >=0.3.2


def main():
    st.title("Network Simulation and Visualization")
    st.caption("Explore regular vs random connectivity and emergent properties.")

    with st.sidebar:
        st.header("Parameters")
        n = st.slider("Nodes (n)", min_value=1, max_value=1000, value=100, step=10)
        m = st.slider("Connections per node (m)", min_value=1, max_value=min(50, n - 1), value=4)

        model = st.selectbox(
            "Model",
            [
                "Watts–Strogatz (regularity)",
                "Random regular (degree = m)",
                "Erdős–Rényi (avg degree ≈ m)",
                "Barabási–Albert (attach m edges)",
            ],
        )

        reg = 0.8
        if model == "Watts–Strogatz (regularity)":
            reg = st.slider("Regularity (0=random, 1=regular)", 0.0, 1.0, 0.8, 0.01)

        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
        physics = st.checkbox("Enable physics layout", value=True)
        dark = st.checkbox("Dark theme", value=True)
        height_px = st.slider("Canvas height (px)", 300, 1000, 650, 50)

        st.divider()
        st.caption("Tip: For WS, ensure m is even for ring lattice.")

    # Generate graph
    try:
        G = generate_graph(model=model, n=n, m=m, regularity=reg, seed=seed)
    except Exception as e:
        st.error(f"Error generating graph: {e}")
        return

    # Metrics
    n_nodes, n_edges, avg_deg, clustering, components, apl = graph_metrics(G)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.subheader("Network Metrics")
        st.metric("Nodes", n_nodes)
        st.metric("Edges", n_edges)
        st.metric("Average degree", f"{avg_deg:.2f}")
        st.metric("Average clustering", f"{clustering:.4f}")
        st.metric("Connected components", components)
        if apl is not None:
            st.metric("Avg path length", f"{apl:.3f}")
        else:
            st.caption("Avg path length unavailable for large or disconnected graphs.")

        # Degree distribution preview
        degs = [d for _, d in G.degree()]
        hist, _ = np.histogram(degs, bins=min(30, int(np.sqrt(max(1, len(degs))))))
        st.bar_chart(hist)

    with col2:
        st.subheader("Visualization")
        html = render_pyvis(G, height_px=height_px, physics=physics, dark=dark)
        st.components.v1.html(html, height=height_px, scrolling=True)


if __name__ == "__main__":
    main()