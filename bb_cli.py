#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════════╗
║         BigBasket Smart Cart — CLI Tool for Daily Use            ║
║                    Your Personal Grocery AI                      ║
╚══════════════════════════════════════════════════════════════════╝

Usage:
    python bb_cli.py recommend "Banana" "Milk" "Bread"
    python bb_cli.py recommend --top 10 "Tomato" "Onion"
    python bb_cli.py customer BB01001
    python bb_cli.py top-products --n 15
    python bb_cli.py rules --min-lift 1.5
    python bb_cli.py stats
    python bb_cli.py interactive

Quick examples:
    python bb_cli.py recommend "Banana" "Amul Full Cream Milk 1L"
    python bb_cli.py stats
    python bb_cli.py interactive
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import click
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from itertools import combinations


# ── Fancy terminal colors ──────────────────────────────────────────────────────
CYAN   = "\033[96m"
YELLOW = "\033[93m"
GREEN  = "\033[92m"
MAGENTA= "\033[95m"
RESET  = "\033[0m"
BOLD   = "\033[1m"
DIM    = "\033[2m"


def cprint(text, color=CYAN):
    print(f"{color}{text}{RESET}")


def print_banner():
    banner = f"""
{CYAN}{BOLD}
 ██████╗ ██╗ ██████╗ ██████╗  █████╗ ███████╗██╗  ██╗███████╗████████╗
 ██╔══██╗██║██╔════╝ ██╔══██╗██╔══██╗██╔════╝██║ ██╔╝██╔════╝╚══██╔══╝
 ██████╔╝██║██║  ███╗██████╔╝███████║███████╗█████╔╝ █████╗     ██║
 ██╔══██╗██║██║   ██║██╔══██╗██╔══██║╚════██║██╔═██╗ ██╔══╝     ██║
 ██████╔╝██║╚██████╔╝██████╔╝██║  ██║███████║██║  ██╗███████╗   ██║
 ╚═════╝ ╚═╝ ╚═════╝ ╚═════╝ ╚═╝  ╚═╝╚══════╝╚═╝  ╚═╝╚══════╝   ╚═╝

                🛒  Smart Cart Prediction CLI  v2.0
               AI-Powered • Daily Use • Personal Grocery AI
{RESET}"""
    print(banner)


# ── Data loading ──────────────────────────────────────────────────────────────
_cache = {}

def load_data():
    if "data" in _cache:
        return _cache["data"]

    base = os.path.dirname(os.path.abspath(__file__))
    raw  = os.path.join(base, "data", "raw")

    try:
        transactions = pd.read_csv(os.path.join(raw, "transactions.csv"))
        order_items  = pd.read_csv(os.path.join(raw, "order_items.csv"))
        products     = pd.read_csv(os.path.join(raw, "products.csv"))
        customers    = pd.read_csv(os.path.join(raw, "customers.csv"))
        transactions["order_date"] = pd.to_datetime(transactions["order_date"])
    except FileNotFoundError:
        cprint("❌ Data not found! Run: python data/synthetic/generate_dataset.py", "\033[91m")
        sys.exit(1)

    # Load rules
    rules_path = os.path.join(base, "models", "association_rules.csv")
    rules = pd.read_csv(rules_path) if os.path.exists(rules_path) else pd.DataFrame()

    # Build co-occurrence index
    cprint("⚡ Building recommendation index...", DIM)
    sample = transactions["order_id"].sample(min(8000, len(transactions)), random_state=42)
    oi = order_items[order_items["order_id"].isin(sample)]
    co_occur = defaultdict(Counter)
    for _, grp in oi.groupby("order_id"):
        items = grp["product_name"].tolist()
        for a, b in combinations(items, 2):
            co_occur[a][b] += 1
            co_occur[b][a] += 1

    _cache["data"] = (transactions, order_items, products, customers, rules, dict(co_occur))
    return _cache["data"]


def get_recommendations(cart, rules, co_occur, top_n=10):
    scores = defaultdict(float)
    cart_set = set(cart)

    for item in cart:
        for nb, cnt in co_occur.get(item, {}).items():
            if nb not in cart_set:
                scores[nb] += cnt

    if not rules.empty:
        for _, rule in rules.iterrows():
            ants = set(rule["antecedents"].split(", "))
            cons = set(rule["consequents"].split(", "))
            overlap = len(ants & cart_set) / max(len(ants), 1)
            if overlap > 0:
                for item in cons:
                    if item not in cart_set:
                        scores[item] += overlap * rule["lift"] * 10

    ranked = sorted(scores.items(), key=lambda x: -x[1])[:top_n]
    max_score = max((s for _, s in ranked), default=1)
    return [(p, round(s / max(max_score, 1e-9), 3)) for p, s in ranked]


# ── CLI Commands ──────────────────────────────────────────────────────────────
@click.group()
def cli():
    """🛒 BigBasket Smart Cart Prediction CLI"""
    pass


@cli.command()
@click.argument("cart_items", nargs=-1, required=True)
@click.option("--top", "-n", default=8, help="Number of recommendations", show_default=True)
def recommend(cart_items, top):
    """Get cart recommendations. Provide product names as arguments."""
    print_banner()
    transactions, order_items, products, customers, rules, co_occur = load_data()

    cprint(f"\n🛒 Your Cart:", YELLOW)
    for item in cart_items:
        prod_row = products[products["product_name"].str.lower() == item.lower()]
        if not prod_row.empty:
            price = prod_row["base_price"].values[0]
            cat   = prod_row["category"].values[0]
            print(f"  {GREEN}✓{RESET} {item:45s} ₹{price:>8.0f}  {DIM}[{cat}]{RESET}")
        else:
            print(f"  {YELLOW}?{RESET} {item:45s} {DIM}[not in catalog — still searching]{RESET}")

    cart = list(cart_items)
    recs = get_recommendations(cart, rules, co_occur, top_n=top)

    cprint(f"\n🔮 AI Recommendations (Top {top}):", CYAN)
    print(f"  {'#':<3} {'Product':<48} {'Match%':<8} {'Price':>8}  {'Category'}")
    print(f"  {'─'*3} {'─'*48} {'─'*8} {'─'*8}  {'─'*20}")

    for i, (product, score) in enumerate(recs, 1):
        pct  = int(score * 100)
        prod_row = products[products["product_name"] == product]
        price = f"₹{prod_row['base_price'].values[0]:.0f}" if not prod_row.empty else "N/A"
        cat   = prod_row["category"].values[0] if not prod_row.empty else ""
        bar_len = int(score * 20)
        bar  = f"{CYAN}{'█' * bar_len}{DIM}{'░' * (20 - bar_len)}{RESET}"
        print(f"  {i:<3} {product:<48} {bar} {pct:>3}%  {price:>8}  {DIM}{cat}{RESET}")

    cart_total = sum(
        products[products["product_name"] == item]["base_price"].values[0]
        for item in cart if len(products[products["product_name"] == item]) > 0
    )
    rec_total = sum(
        products[products["product_name"] == p]["base_price"].values[0]
        for p, _ in recs if len(products[products["product_name"] == p]) > 0
    )

    print(f"\n  {YELLOW}💰 Cart Total (current):    ₹{cart_total:.0f}{RESET}")
    print(f"  {DIM}💡 Add all recommendations: ₹{rec_total:.0f} more{RESET}\n")


@cli.command()
@click.argument("customer_id")
def customer(customer_id):
    """Show customer profile and top purchases."""
    transactions, order_items, products, customers, rules, co_occur = load_data()

    cust = customers[customers["customer_id"] == customer_id]
    if cust.empty:
        cprint(f"❌ Customer {customer_id} not found.", "\033[91m")
        return

    cust_orders = transactions[transactions["customer_id"] == customer_id]
    cust_items  = order_items[order_items["order_id"].isin(cust_orders["order_id"])]
    info = cust.iloc[0]

    print_banner()
    cprint(f"👤 Customer Profile: {customer_id}", YELLOW)
    print(f"  City         : {info['city']}")
    print(f"  Tier         : {YELLOW}{info['loyalty_tier']}{RESET}")
    print(f"  Age Group    : {info['age_group']}")
    print(f"  BB Star      : {'⭐ Yes' if info['has_bb_star'] else 'No'}")
    print(f"  Since        : {info['registered_since']}")
    print()
    cprint(f"📊 Purchase Summary", CYAN)
    print(f"  Total Orders : {len(cust_orders):,}")
    print(f"  Total Spent  : ₹{cust_orders['order_total'].sum():,.0f}")
    print(f"  Avg Order    : ₹{cust_orders['order_total'].mean():,.0f}")

    cprint(f"\n🏷️ Top 10 Most Purchased Products:", CYAN)
    top = cust_items["product_name"].value_counts().head(10)
    for i, (prod, cnt) in enumerate(top.items(), 1):
        print(f"  {i:2d}. {prod:<45} {cnt}x")


@cli.command()
@click.option("--n", default=15, help="Number of products to show")
def top_products(n):
    """Show top products by order count."""
    transactions, order_items, products, customers, rules, co_occur = load_data()
    print_banner()
    cprint(f"🏆 Top {n} Products by Orders", YELLOW)

    top = (order_items.groupby("product_name")
           .agg(orders=("order_id","nunique"),
                revenue=("final_price","sum"),
                avg_price=("unit_price","mean"))
           .reset_index()
           .sort_values("revenue", ascending=False)
           .head(n))

    print(f"\n  {'#':<3} {'Product':<45} {'Orders':>8} {'Revenue':>12} {'Avg Price':>10}")
    print(f"  {'─'*3} {'─'*45} {'─'*8} {'─'*12} {'─'*10}")
    for i, row in top.iterrows():
        print(f"  {list(top.index).index(i)+1:<3} {row['product_name']:<45} "
              f"{int(row['orders']):>8,} ₹{row['revenue']:>10,.0f} ₹{row['avg_price']:>8.0f}")


@cli.command()
@click.option("--min-lift", default=1.2, help="Minimum lift value")
@click.option("--n", default=20, help="Number of rules to show")
def rules(min_lift, n):
    """Display association rules."""
    transactions, order_items, products, customers, rules_df, co_occur = load_data()
    print_banner()

    if rules_df.empty:
        cprint("⚠️  No rules found. Run: python train_pipeline.py", YELLOW)
        return

    filtered = rules_df[rules_df["lift"] >= min_lift].head(n)
    cprint(f"🔗 Top {len(filtered)} Association Rules (lift ≥ {min_lift})", YELLOW)
    print(f"\n  {'Antecedent → Consequent':<70} {'Lift':>6} {'Conf':>6} {'Sup':>6}")
    print(f"  {'─'*70} {'─'*6} {'─'*6} {'─'*6}")
    for _, rule in filtered.iterrows():
        rule_str = f"{rule['antecedents']} → {rule['consequents']}"
        if len(rule_str) > 68:
            rule_str = rule_str[:65] + "..."
        print(f"  {rule_str:<70} {rule['lift']:>6.2f} {rule['confidence']:>6.2f} {rule['support']:>6.3f}")


@cli.command()
def stats():
    """Show store analytics summary."""
    transactions, order_items, products, customers, rules, co_occur = load_data()
    print_banner()
    cprint("📊 Store Analytics Summary", YELLOW)

    print(f"""
  {CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
  {BOLD}Transactions{RESET}
    Total Orders      : {len(transactions):>12,}
    Unique Customers  : {transactions['customer_id'].nunique():>12,}
    Total Revenue     : ₹{transactions['order_total'].sum():>11,.0f}
    Avg Order Value   : ₹{transactions['order_total'].mean():>11,.0f}
    Avg Basket Size   : {transactions['basket_size'].mean():>12.2f} items
    Date Range        : {transactions['order_date'].min().date()} → {transactions['order_date'].max().date()}

  {CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
  {BOLD}Catalog{RESET}
    Total Products    : {order_items['product_name'].nunique():>12,}
    Categories        : {order_items['category'].nunique():>12,}
    Top Category      : {order_items['category'].value_counts().index[0]:>20}
    Top Product       : {order_items['product_name'].value_counts().index[0]:>20}

  {CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
  {BOLD}Customers{RESET}
    Top City          : {transactions['city'].value_counts().index[0]:>20}
    Top Payment       : {transactions['payment_method'].value_counts().index[0]:>20}
    Express Orders    : {transactions['is_express'].sum():>12,} ({transactions['is_express'].mean()*100:.0f}%)
    Association Rules : {len(rules):>12,}
  {CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━{RESET}
""")


@cli.command()
def interactive():
    """🛒 Interactive cart builder (type items, get live recommendations)."""
    transactions, order_items, products, customers, rules, co_occur = load_data()
    print_banner()
    cprint("🛒 Interactive Smart Cart Builder", YELLOW)
    cprint("   Type product names one by one. Press Enter after each.", DIM)
    cprint("   Commands: 'done' to finish, 'clear' to reset, 'quit' to exit\n", DIM)

    cart = []
    all_products = sorted(order_items["product_name"].unique().tolist())

    while True:
        prompt = f"{CYAN}Add item ({len(cart)} in cart)>{RESET} "
        try:
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            cprint("\n👋 Goodbye!", GREEN)
            break

        if user_input.lower() in ("quit", "exit", "q"):
            cprint("👋 Goodbye!", GREEN)
            break
        elif user_input.lower() in ("done", "d"):
            if not cart:
                cprint("  Cart is empty!", YELLOW)
                continue
            recs = get_recommendations(cart, rules, co_occur, top_n=10)
            cprint(f"\n🔮 Your personalized recommendations:", CYAN)
            for i, (p, s) in enumerate(recs, 1):
                prod_row = products[products["product_name"] == p]
                price = f"₹{prod_row['base_price'].values[0]:.0f}" if not prod_row.empty else ""
                print(f"  {i:2d}. {p:<45} {int(s*100):>3}%  {price}")
            print()
        elif user_input.lower() == "clear":
            cart = []
            cprint("  🗑️  Cart cleared!", GREEN)
        elif not user_input:
            continue
        else:
            # Fuzzy search
            matches = [p for p in all_products if user_input.lower() in p.lower()]
            if not matches:
                cprint(f"  ❌ No product matching '{user_input}' found.", "\033[91m")
                similar = [p for p in all_products if any(
                    w in p.lower() for w in user_input.lower().split()
                )][:5]
                if similar:
                    cprint("  Did you mean:", DIM)
                    for s in similar:
                        print(f"    • {s}")
            elif len(matches) == 1:
                item = matches[0]
                cart.append(item)
                prod_row = products[products["product_name"] == item]
                price = f"₹{prod_row['base_price'].values[0]:.0f}" if not prod_row.empty else ""
                cprint(f"  {GREEN}✓ Added: {item} {price}{RESET}")
                # Quick preview
                recs = get_recommendations(cart, rules, co_occur, top_n=3)
                if recs:
                    cprint(f"  💡 Quick recs: {', '.join(p for p, _ in recs)}", DIM)
            else:
                cprint(f"  Multiple matches for '{user_input}':", DIM)
                for i, m in enumerate(matches[:8], 1):
                    print(f"    {i}. {m}")
                try:
                    choice = int(input(f"  Pick (1-{min(8, len(matches))}): ").strip())
                    if 1 <= choice <= min(8, len(matches)):
                        item = matches[choice - 1]
                        cart.append(item)
                        cprint(f"  {GREEN}✓ Added: {item}{RESET}")
                except (ValueError, EOFError):
                    cprint("  Skipped.", DIM)


if __name__ == "__main__":
    cli()
