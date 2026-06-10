"""
╔══════════════════════════════════════════════════════════════════════════════╗
║                    BIGBASKET CART PREDICTION — TRAINING PIPELINE            ║
║                         End-to-End Model Training Script                    ║
╚══════════════════════════════════════════════════════════════════════════════╝

Run:
    python train_pipeline.py --data_dir data/raw --output_dir models/

Steps:
    1. Load & validate datasets
    2. Feature engineering
    3. Train Association Rule Miner (FP-Growth)
    4. Train Collaborative Filter (SVD)
    5. Train Next Basket Predictor (Sequential)
    6. Train Hybrid Ensemble
    7. Evaluate & save all artifacts
"""

import os, sys, json, pickle, argparse, warnings
from datetime import datetime

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models.cart_predictor import (
    AssociationRuleMiner, CollaborativeFilter,
    NextBasketPredictor, HybridCartPredictor
)
from src.features.feature_engineering import (
    build_customer_features, build_product_features,
    build_temporal_features, compute_reorder_labels
)


# ── Utility ───────────────────────────────────────────────────────────────────
def banner(title: str):
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def load_data(data_dir: str):
    banner("📂 LOADING DATASETS")
    transactions  = pd.read_csv(os.path.join(data_dir, "transactions.csv"))
    order_items   = pd.read_csv(os.path.join(data_dir, "order_items.csv"))
    products      = pd.read_csv(os.path.join(data_dir, "products.csv"))
    customers     = pd.read_csv(os.path.join(data_dir, "customers.csv"))

    print(f"  Transactions : {len(transactions):>8,} rows")
    print(f"  Order items  : {len(order_items):>8,} rows")
    print(f"  Products     : {len(products):>8,}")
    print(f"  Customers    : {len(customers):>8,}")
    return transactions, order_items, products, customers


def evaluate_arm(arm: AssociationRuleMiner, order_items: pd.DataFrame,
                 transactions: pd.DataFrame, n_test: int = 1000) -> dict:
    """Evaluate ARM recommendations using precision@k and recall@k."""
    banner("📊 EVALUATING ARM MODEL")
    transactions = transactions.copy()
    transactions["order_date"] = pd.to_datetime(transactions["order_date"])
    transactions = transactions.sort_values(["customer_id", "order_date"])

    customer_orders = order_items.merge(
        transactions[["order_id", "customer_id"]], on="order_id"
    )

    prec_scores, rec_scores = [], []
    sample_customers = transactions["customer_id"].unique()[:n_test]

    for cid in sample_customers:
        cust_orders = transactions[transactions["customer_id"] == cid]["order_id"].tolist()
        if len(cust_orders) < 2:
            continue
        last_order = cust_orders[-1]
        prev_orders = cust_orders[:-1]

        hist_items = customer_orders[
            customer_orders["order_id"].isin(prev_orders)
        ]["product_name"].tolist()
        target_items = set(customer_orders[
            customer_orders["order_id"] == last_order
        ]["product_name"].tolist())

        if not hist_items or not target_items:
            continue

        recs = [r["product"] for r in arm.recommend(hist_items[-5:], top_n=10)]
        hits = len(set(recs) & target_items)
        prec_scores.append(hits / len(recs) if recs else 0)
        rec_scores.append(hits / len(target_items) if target_items else 0)

    metrics = {
        "precision@10": round(np.mean(prec_scores), 4),
        "recall@10":    round(np.mean(rec_scores), 4),
        "f1@10":        round(2 * np.mean(prec_scores) * np.mean(rec_scores)
                              / (np.mean(prec_scores) + np.mean(rec_scores) + 1e-9), 4),
    }
    for k, v in metrics.items():
        print(f"  {k:<20}: {v:.4f}")
    return metrics


def main(args):
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "artifacts"), exist_ok=True)

    # 1. Load data
    transactions, order_items, products, customers = load_data(args.data_dir)

    # 2. Feature Engineering
    banner("🔧 FEATURE ENGINEERING")
    customer_feats = build_customer_features(transactions, order_items)
    product_feats  = build_product_features(order_items, transactions)
    transactions_temporal = build_temporal_features(transactions)
    print(f"  Customer features : {customer_feats.shape[1]} features × {len(customer_feats)} customers")
    print(f"  Product features  : {product_feats.shape[1]} features × {len(product_feats)} products")

    customer_feats.to_csv(os.path.join(args.output_dir, "customer_features.csv"))
    product_feats.to_csv(os.path.join(args.output_dir, "product_features.csv"))
    transactions_temporal.to_csv(os.path.join(args.output_dir, "transactions_temporal.csv"), index=False)
    print("  ✅ Feature CSVs saved")

    # 3. Train Hybrid Model
    banner("🤖 TRAINING HYBRID CART PREDICTOR")
    model = HybridCartPredictor()
    model.fit(transactions, order_items, customers)

    # 4. Evaluate ARM
    metrics = evaluate_arm(model.arm, order_items, transactions)

    # 5. Save association rules
    banner("💾 SAVING ARTIFACTS")
    rules_path = os.path.join(args.output_dir, "association_rules.csv")
    if model.arm.rules_ is not None and not model.arm.rules_.empty:
        model.arm.rules_.to_csv(rules_path, index=False)
        print(f"  Saved {len(model.arm.rules_):,} association rules → {rules_path}")

    # 6. Save model
    model_path = os.path.join(args.output_dir, "artifacts", f"hybrid_model_{run_id}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved hybrid model → {model_path}")

    # 7. Save run metadata
    meta = {
        "run_id": run_id,
        "n_transactions": len(transactions),
        "n_order_items": len(order_items),
        "n_customers": len(customers),
        "n_products": len(products),
        "n_association_rules": len(model.arm.rules_) if model.arm.rules_ is not None else 0,
        "arm_metrics": metrics,
        "timestamp": datetime.now().isoformat(),
    }
    meta_path = os.path.join(args.output_dir, "artifacts", f"run_meta_{run_id}.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Saved run metadata → {meta_path}")

    banner("✅ TRAINING COMPLETE")
    print(f"  Run ID  : {run_id}")
    print(f"  P@10    : {metrics['precision@10']}")
    print(f"  R@10    : {metrics['recall@10']}")
    print(f"  F1@10   : {metrics['f1@10']}")

    # Quick demo
    banner("🛒 DEMO: RECOMMENDATIONS")
    demo_user = transactions["customer_id"].iloc[0]
    demo_tier = customers[customers["customer_id"] == demo_user]["loyalty_tier"].values[0]
    demo_cart = ["Banana", "Amul Full Cream Milk 1L", "Britannia Brown Bread"]
    print(f"  User  : {demo_user} ({demo_tier})")
    print(f"  Cart  : {demo_cart}")
    recs = model.recommend(demo_user, demo_cart, tier=demo_tier, top_n=8)
    print("\n  📦 Recommended Items:")
    for r in recs:
        print(f"    {r['rank']:2d}. {r['product']:<45} score={r['ensemble_score']:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BigBasket Cart Prediction Training Pipeline")
    parser.add_argument("--data_dir",    default="data/raw",    help="Directory with raw CSV files")
    parser.add_argument("--output_dir",  default="models",      help="Output directory for artifacts")
    args = parser.parse_args()
    main(args)
