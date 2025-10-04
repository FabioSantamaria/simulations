import streamlit as st
import networkx as nx
import numpy as np
from pyvis.network import Network
from typing import Tuple


st.set_page_config(page_title="Network Simulator", layout="wide")


def generate_graph(model: str, n: int, m: int, regularity: float, seed: int, initial_n: int | None = None) -> nx.Graph:
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

    if model == "Preferential growth (1 edge/time)":
        n0 = initial_n if initial_n is not None else max(2, min(10, n))
        m0 = max(0, min(m, (n0 * (n0 - 1)) // 2))
        return preferential_growth_graph(n_total=n, n0=n0, m0=m0, seed=seed)

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


def render_pyvis(G: nx.Graph, height_px: int, physics: bool, dark: bool, pos: dict | None = None, fixed_length_px: int | None = None, canvas_size_px: int | None = None, zoom_scale: float | None = None) -> str:
    bg = "#0f1116" if dark else "#ffffff"
    font = "#e0e0e0" if dark else "#2b2b2b"
    if canvas_size_px is not None:
        net = Network(height=f"{canvas_size_px}px", width=f"{canvas_size_px}px", notebook=False, bgcolor=bg, font_color=font)
    else:
        net = Network(height=f"{height_px}px", width="100%", notebook=False, bgcolor=bg, font_color=font)

    # Configure physics/layout: optionally enforce a fixed edge length and non-overlap
    if fixed_length_px is not None:
        # Use physics with a target spring length and node distance to reduce overlap
        net.set_options(f"""
{{
  "physics": {{
    "enabled": true,
    "solver": "repulsion",
    "repulsion": {{
      "nodeDistance": {fixed_length_px},
      "springLength": {fixed_length_px},
      "springConstant": 0.01,
      "damping": 0.09
    }},
    "minVelocity": 0.1,
    "stabilization": {{ "enabled": true, "fit": false, "iterations": 40 }}
  }},
  "layout": {{ "improvedLayout": false }},
  "edges": {{ "length": {fixed_length_px} }},
  "interaction": {{ "hover": true }}
}}
        """)
        scale = None
    else:
        # If explicit positions are provided, fix nodes there and disable physics
        if pos is not None:
            net.set_options("""
{
  "physics": { "enabled": false },
  "layout": { "improvedLayout": false },
  "interaction": { "hover": true }
}
            """)
            # Scale positions to canvas coordinates
            base_size = canvas_size_px if canvas_size_px is not None else height_px
            scale = max(300, min(1200, base_size - 80))
        else:
            # Improve layout stability for medium/large graphs (client-side physics)
            if physics:
                net.barnes_hut(gravity=-5000, central_gravity=0.2, spring_length=120, spring_strength=0.05, damping=0.9)
                net.set_options("""
{
  "layout": { "improvedLayout": false },
  "interaction": { "hover": true }
}
                """)
            else:
                net.set_options("""
{
  "physics": { "enabled": false },
  "layout": { "improvedLayout": false },
  "interaction": { "hover": true }
}
                """)
            scale = None

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
        if pos is not None and node in pos and fixed_length_px is None:
            x = int(pos[node][0] * (scale or height_px))
            y = int(pos[node][1] * (scale or height_px))
            net.add_node(
                node,
                label=str(node),
                title=f"Node {node} | degree {d}",
                color=color_for_degree(d),
                x=x,
                y=y,
                physics=False,
            )
        else:
            net.add_node(
                node,
                label=str(node),
                title=f"Node {node} | degree {d}",
                color=color_for_degree(d),
            )

    for u, v in G.edges():
        if fixed_length_px is not None:
            net.add_edge(u, v, length=fixed_length_px)
        else:
            net.add_edge(u, v)

    # Return HTML string to embed in Streamlit, with optional initial zoom
    html = net.generate_html()  # pyvis >=0.3.2
    if zoom_scale is not None:
        try:
            scale_js = max(0.1, min(5.0, float(zoom_scale) / 100.0))
            html = html.replace(
                "</body>",
                f"<script>try{{network.moveTo({{scale:{scale_js}}});}}catch(e){{}}</script></body>",
            )
        except Exception:
            pass
    return html


def preferential_growth_graph(n_total: int, n0: int, m0: int, seed: int) -> nx.Graph:
    rng = np.random.default_rng(seed)
    n0 = max(1, min(n0, n_total))
    G = nx.Graph()
    G.add_nodes_from(range(n0))

    # Initialize with m0 random edges among initial nodes (no duplicates/self-loops)
    possible = [(i, j) for i in range(n0) for j in range(i + 1, n0)]
    rng.shuffle(possible)
    for (u, v) in possible[:min(m0, len(possible))]:
        G.add_edge(u, v)

    # Grow: each new node connects to one existing node proportional to degree
    for new_node in range(n0, n_total):
        G.add_node(new_node)
        existing = list(range(new_node))
        degs = np.array([G.degree(u) for u in existing], dtype=float)
        total = float(degs.sum())
        if total <= 0.0:
            target = int(rng.choice(existing))
        else:
            probs = degs / total
            target = int(rng.choice(existing, p=probs))
        G.add_edge(new_node, target)

    return G


def compute_layout(G: nx.Graph, prev_pos: dict | None, seed: int) -> dict:
    # Compute spring layout, reusing previous positions for continuity
    try:
        pos = nx.spring_layout(G, seed=seed, pos=prev_pos, iterations=30)
    except Exception:
        pos = nx.spring_layout(G, seed=seed)
    return pos


def build_initial_growth_graph(n0: int, m0: int, seed: int) -> nx.Graph:
    # Create only the initial graph with n0 nodes and m0 edges
    rng = np.random.default_rng(seed)
    n0 = max(1, n0)
    G = nx.Graph()
    G.add_nodes_from(range(n0))
    possible = [(i, j) for i in range(n0) for j in range(i + 1, n0)]
    rng.shuffle(possible)
    for (u, v) in possible[:min(m0, len(possible))]:
        G.add_edge(u, v)
    return G


def add_one_node_preferential(G: nx.Graph, new_node_id: int) -> None:
    # Add a single node that attaches to one existing node with probability ∝ degree
    G.add_node(new_node_id)
    existing = [u for u in G.nodes() if u != new_node_id]
    degs = np.array([G.degree(u) for u in existing], dtype=float)
    # Use degree + bias constant; clamp negatives
    bias_c = float(st.session_state.get("growth_bias_c", 0.0)) if "growth_bias_c" in st.session_state else 0.0
    weights = np.clip(degs + bias_c, a_min=0.0, a_max=None)
    total = float(weights.sum())
    rng = np.random.default_rng()
    if total <= 0.0:
        target = int(rng.choice(existing))
    else:
        probs = weights / total
        target = int(rng.choice(existing, p=probs))
    G.add_edge(new_node_id, target)


def main():
    st.title("Network Simulation and Visualization")
    st.caption("Explore regular vs random connectivity and emergent properties.")

    # For interactive growth mode buttons
    init_clicked = False
    add_clicked = False

    with st.sidebar:
        st.header("Parameters")
        n = st.slider("Final nodes (n)", min_value=1, max_value=1000, value=100, step=10)

        model = st.selectbox(
            "Model",
            [
                "Watts–Strogatz (regularity)",
                "Random regular (degree = m)",
                "Erdős–Rényi (avg degree ≈ m)",
                "Barabási–Albert (attach m edges)",
                "Preferential growth (1 edge/time)",
            ],
        )

        # Model-specific controls
        initial_n = None
        if model == "Preferential growth (1 edge/time)":
            initial_n = st.slider("Initial nodes (n0)", min_value=1, max_value=min(100, n), value=min(10, n))
            max_m0 = (initial_n * (initial_n - 1)) // 2
            m = st.slider("Initial edges (m0)", min_value=0, max_value=max_m0, value=min(10, max_m0))
            c1, c2 = st.columns(2)
            with c1:
                init_clicked = st.button("Initialize growth", use_container_width=True)
            with c2:
                add_clicked = st.button("Add one node", use_container_width=True)
        elif model == "Watts–Strogatz (regularity)":
            m = st.slider("Nearest neighbors (m)", min_value=2, max_value=min(50, n - 1), value=4)
        elif model in ("Random regular (degree = m)", "Barabási–Albert (attach m edges)"):
            m = st.slider("Connections per node (m)", min_value=1, max_value=min(50, n - 1), value=4)
        else:
            m = st.slider("Target avg degree (m)", min_value=1, max_value=min(50, n - 1), value=4)

        reg = 0.8
        if model == "Watts–Strogatz (regularity)":
            reg = st.slider("Regularity (0=random, 1=regular)", 0.0, 1.0, 0.8, 0.01)

        seed = st.number_input("Random seed", min_value=0, max_value=1_000_000, value=42, step=1)
        physics = st.checkbox("Enable physics layout", value=True)
        dark = st.checkbox("Dark theme", value=True)
        height_px = st.slider("Canvas height (px)", 300, 1000, 650, 50)

        st.divider()
        if model == "Watts–Strogatz (regularity)":
            st.caption("Tip: For WS, ensure m is even for ring lattice.")
        if model == "Preferential growth (1 edge/time)":
            st.caption("Each new node connects to one existing node with probability ∝ degree.")

    # Generate or update graph
    try:
        if model == "Preferential growth (1 edge/time)":
            # Initialize growth state
            if init_clicked or "G_growth" not in st.session_state:
                st.session_state["G_growth"] = build_initial_growth_graph(n0=initial_n, m0=m, seed=seed)
                st.session_state["growth_next"] = initial_n
                st.session_state["final_n"] = n
                # initial layout
                st.session_state["pos_growth"] = compute_layout(st.session_state["G_growth"], prev_pos=None, seed=seed)

            # Add one node if requested and capacity remains
            if add_clicked:
                next_id = st.session_state.get("growth_next", initial_n)
                final_n = st.session_state.get("final_n", n)
                if next_id >= final_n:
                    st.info("Reached target number of nodes.")
                else:
                    add_one_node_preferential(st.session_state["G_growth"], new_node_id=next_id)
                    # recompute layout with previous positions for smoothness
                    st.session_state["pos_growth"] = compute_layout(st.session_state["G_growth"], prev_pos=st.session_state.get("pos_growth"), seed=seed)
                    st.session_state["growth_next"] = next_id + 1

            G = st.session_state["G_growth"]
        else:
            G = generate_graph(model=model, n=n, m=m, regularity=reg, seed=seed, initial_n=initial_n)
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

        # Degree distribution (exact counts per integer degree)
        degs = [d for _, d in G.degree()]
        if degs:
            max_d = max(degs)
            counts = np.bincount(degs, minlength=max_d + 1)
            import pandas as pd
            df_deg = pd.DataFrame({"degree": np.arange(len(counts)), "count": counts})
            st.bar_chart(df_deg.set_index("degree"))
        else:
            st.caption("No degree data available.")

        # Model parameters summary for clarity
        st.divider()
        if model == "Watts–Strogatz (regularity)":
            k_eff = m if m % 2 == 0 else max(2, m - 1)
            p_eff = float(max(0.0, min(1.0, 1.0 - reg)))
            st.caption(f"WS effective k={k_eff}, rewiring p={p_eff:.2f} (degrees vary around k when p>0)")
        elif model == "Random regular (degree = m)":
            m_eff = m
            if (n * m) % 2 == 1:
                m_eff = m + 1 if m + 1 < n else m - 1
            st.caption(f"Random regular: enforced degree d={m_eff} for all nodes")
        elif model == "Erdős–Rényi (avg degree ≈ m)":
            p_eff = float(m) / float(max(1, n - 1))
            p_eff = float(max(0.0, min(1.0, p_eff)))
            st.caption(f"ER: edge probability p≈{p_eff:.4f}; expected average degree ≈ m, not exact")
        elif model == "Barabási–Albert (attach m edges)":
            st.caption("BA: new nodes attach m edges; degrees are skewed with hubs; min degree ≥ m")
        elif model == "Preferential growth (1 edge/time)":
            st.caption("Preferential growth: start with n0 nodes, m0 edges; each new node attaches to an existing node with probability proportional to its degree.")

    with col2:
        st.subheader("Visualization")
        if model == "Preferential growth (1 edge/time)":
            # Optional fixed edge length controls
            fix_len = st.checkbox("Fix edge length during growth", value=True)
            edge_len_px = st.slider("Edge length (px)", 50, 300, 120, 10)
            # Zoom controls (scales the view without changing positions)
            zoom_pct = st.slider("Zoom (%)", 50, 300, 100, 10)
            # Bias constant control for attachment probability
            bias_c = st.number_input("Attachment bias constant (c)", min_value=0.0, value=0.0, step=0.5)
            st.session_state["growth_bias_c"] = float(bias_c)
            html = render_pyvis(
                G,
                height_px=height_px,
                physics=not fix_len,
                dark=dark,
                pos=st.session_state.get("pos_growth"),
                fixed_length_px=edge_len_px if fix_len else None,
                canvas_size_px=None,
                zoom_scale=zoom_pct,
            )
        else:
            html = render_pyvis(G, height_px=height_px, physics=physics, dark=dark, pos=None, fixed_length_px=None, canvas_size_px=None, zoom_scale=None)
        st.components.v1.html(html, height=height_px, scrolling=True)


if __name__ == "__main__":
    main()