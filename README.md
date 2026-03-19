# Applied-active-inference
Warehouse experiments with Kaggle datasets

## Setup

```bash
pip install -r requirements.txt
```

## Downloading Datasets

This project uses four Kaggle datasets. To download them all:

```bash
python download_datasets.py
```

You need Kaggle credentials. Either:
1. Place your `kaggle.json` in `~/.kaggle/` (download from https://www.kaggle.com/settings → API → Create New Token)
2. Or set environment variables: `KAGGLE_USERNAME` and `KAGGLE_KEY`

### Datasets
- [Logistics and Supply Chain](https://www.kaggle.com/datasets/datasetengineer/logistics-and-supply-chain-dataset)
- [Car Sales Report](https://www.kaggle.com/datasets/missionjee/car-sales-report)
- [Inventory Management (Grocery)](https://www.kaggle.com/datasets/mustofaahmad/inventory-management-grocery-industry)
- [Retail Store Inventory Forecasting](https://www.kaggle.com/datasets/anirudhchauhan/retail-store-inventory-forecasting-dataset)

## Project Structure

| File | Description |
|------|-------------|
| `supply_chain_pomdp.ipynb` | Neural-net world model + active inference agent on logistics time-series (32k rows) |
| `grocery_inventory_ai.ipynb` | Parametric simulator + active inference agent on grocery snapshot (1000 SKUs) |
| `src/applied_active_inference/grocery.py` | Simulator, preferences, agents for grocery inventory |
| `src/applied_active_inference/train.py` | SupplyChainMLP observation model |
| `src/applied_active_inference/supply_chain_dataset.py` | Data loading for logistics dataset |

## Results: Grocery Inventory AI

Active inference agent vs classical reorder-point (s,Q) baseline across all 1000 SKUs,
90-day simulation, 3 random seeds each.

| Metric | Baseline | AI Agent | Delta |
|--------|----------|----------|-------|
| Fill rate | 60.9% | **65.2%** | +4.3pp |
| Stockout days (of 90) | 35.2 | **31.3** | -3.9 |
| Total orders placed | 85.5 | **50.3** | -41% |
| Lost sales (units) | 828 | **733** | -11% |
| Avg inventory (units) | 44 | 46 | +3% |

- AI agent achieves better fill rate on **757 / 1000 SKUs**
- Every product category sees improvement (+3.5 to +5.4 pp fill rate)
- Largest gains: Frozen (+5.4pp), Bakery (+5.2pp)
- AI orders proactively via EFE-based planning rather than reacting at a fixed threshold,
  resulting in 41% fewer orders at higher service levels

## Roadmap

### Done
- [x] **Logistics POMDP** (`supply_chain_pomdp.ipynb`): neural-net step function
      learning P(s'|s,a) from 32k hourly time-series, MCMC trajectory simulation,
      homeostatic preferences, active inference agent vs behavioral baseline
- [x] **Grocery inventory AI** (`grocery_inventory_ai.ipynb`): parametric inventory
      simulator from 1000-SKU snapshot, order-up-to preference target
      (S = d*(L+R) + z*sigma*sqrt(L+R)), active inference agent with adaptive
      planning horizon and analytical step distribution, comparison vs classical
      reorder-point (s,Q) baseline

### In Progress
- [ ] **Mahault**: Active inference model on grocery dataset — binary daily order/hold
      decision, EFE-based planning, compare against reorder-point baseline
- [ ] **Jasmine**: Basic statistical / Monte Carlo baseline on grocery dataset —
      keep within demand std bounds, optimize stock levels given shipment uncertainty

### Next: Model Improvements
- [ ] Add cost model (holding cost + stockout penalty + ordering cost) for dollar-value comparison
- [ ] Tune service-level z-score per ABC class (A-class gets higher z = less stockout risk)
- [ ] Softmax policy selection with temperature parameter instead of argmin
- [ ] Compare against: Thompson sampling, UCB, simple neural net baseline

### Next: Scaling
- [ ] Multi-warehouse network: each warehouse = node, compose step functions,
      coupling = transfer flows between nodes
- [ ] Closed-form stock-out probability (analytical vs Monte Carlo for the single-warehouse case)

### Next: Learning
- [ ] Learn preference parameters from data (computational phenotyping —
      inverse active inference to infer what ranking of outcomes explains observed behavior)
- [ ] Add proper epistemic actions (explore disruption states before committing)

### Logistics Notebook Specific
- [ ] Integrate existing `SupplyChainMLP` checkpoint as observation model
- [ ] Validate step function on held-out sequences before plugging in policy
- [ ] Infer disruption severity from state transitions
