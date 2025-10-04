# Network Simulation and Visualization (Streamlit)

An interactive app to explore different network models and visualize their structures. Adjust the number of nodes, connections per node, and a regularity parameter to move between highly regular lattices and random networks.

## Features

- Watts–Strogatz model with a "regularity" slider (rewiring probability)
- Random regular graphs with fixed degree `m`
- Erdős–Rényi graphs tuned to target average degree `m`
- Barabási–Albert scale-free networks
- PyVis-based interactive visualization embedded in Streamlit
- Live metrics: nodes, edges, average degree, clustering coefficient, components, and average path length (when feasible)

## Setup

```bash
pip install -r requirements.txt
```

## Run Locally

```bash
streamlit run app.py
```

Open the local URL shown in the terminal (usually `http://localhost:8501`).

## Usage Tips

- For the Watts–Strogatz model, `m` should be even to form a symmetric ring lattice.
- The "Regularity" slider maps to rewiring probability as `p = 1 - regularity`.
- Large graphs can be heavy to lay out. Disable physics to keep the layout static.

## Deploy to Streamlit Cloud

1. Push this folder to a Git repository (e.g., GitHub).
2. In Streamlit Cloud, create a new app and select your repo.
3. Set the entry point to `app.py`.
4. Ensure `requirements.txt` is included.

## Deploy to Your Own Server

- Basic: Run `streamlit run app.py` behind a reverse proxy (e.g., Nginx) and expose the chosen port.
- Headless mode (optional):

```bash
streamlit run app.py --server.headless true --server.port 8501 --server.address 0.0.0.0
```

## License

This project is provided as-is for educational and research use.