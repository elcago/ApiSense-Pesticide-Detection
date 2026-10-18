import numpy as np
import pandas as pd
import torch
from train import WAMNet, build_sequences, fit_scalers
from train import FORAGING_COLS, ACOUSTIC_COLS, LABEL_COL, SORT_COLS

torch.set_num_threads(1)

DATA_CSV = "data/ApiSense_v17.csv"
WEIGHTS = "checkpoints/wam_net_best.pt"
N_STEPS = 50
FEATURE_COLS = FORAGING_COLS + ACOUSTIC_COLS
N_FORAGING = len(FORAGING_COLS)


def integrated_gradients(model, foraging, acoustic, weather, base_foraging, base_acoustic):
    grad_f = torch.zeros_like(foraging)
    grad_a = torch.zeros_like(acoustic)

    for alpha in torch.linspace(0, 1, N_STEPS):
        f = (base_foraging + alpha * (foraging - base_foraging)).requires_grad_(True)
        a = (base_acoustic + alpha * (acoustic - base_acoustic)).requires_grad_(True)
        gf, ga = torch.autograd.grad(model(f, a, weather).sum(), [f, a])
        grad_f += gf
        grad_a += ga

    ig_f = (foraging - base_foraging) * grad_f / N_STEPS
    ig_a = (acoustic - base_acoustic) * grad_a / N_STEPS
    return torch.cat([ig_f, ig_a], dim=-1)


if __name__ == "__main__":
    import pathlib
    pathlib.Path("results").mkdir(exist_ok=True)

    df = pd.read_csv(DATA_CSV)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = WAMNet().to(device)
    model.load_state_dict(torch.load(WEIGHTS, map_location=device))
    model.eval()

    colony_ids = df["colony_id"].unique()
    scalers = fit_scalers(df, colony_ids)
    data = {cid: build_sequences(df, cid, scalers) for cid in colony_ids}

    treated_ids = [c for c in colony_ids if data[c][3] == 1]
    control_ids = [c for c in colony_ids if data[c][3] == 0]

    foraging = torch.stack([data[c][0] for c in treated_ids]).to(device)
    acoustic = torch.stack([data[c][1] for c in treated_ids]).to(device)
    weather = torch.stack([data[c][2] for c in treated_ids]).to(device)

    base_foraging = torch.stack([data[c][0] for c in control_ids]).mean(dim=0, keepdim=True).to(device)
    base_acoustic = torch.stack([data[c][1] for c in control_ids]).mean(dim=0, keepdim=True).to(device)
    base_foraging = base_foraging.expand_as(foraging)
    base_acoustic = base_acoustic.expand_as(acoustic)

    ig = integrated_gradients(model, foraging, acoustic, weather, base_foraging, base_acoustic)
    attribution = ig.abs().mean(dim=0).detach().cpu().numpy()

    day_num = (
        df[df["colony_id"] == treated_ids[0]]
        .sort_values(SORT_COLS)["day_num"]
        .values
    )

    temporal = pd.DataFrame(attribution, columns=FEATURE_COLS)
    temporal["day_num"] = day_num
    temporal = temporal.groupby("day_num")[FEATURE_COLS].sum()
    temporal = temporal.div(temporal.sum(axis=1), axis=0)
    temporal.to_csv("results/temporal_attribution.csv")

    importance = attribution.sum(axis=0)
    importance = importance / importance.sum()
    importance_df = (
        pd.DataFrame({"feature": FEATURE_COLS, "importance": importance})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    importance_df.to_csv("results/feature_importance.csv", index=False)

    print("Top feature per day:")
    for day, row in temporal.iterrows():
        print(f"  Day {int(day):>2}: {row.idxmax():<20} {row.max():.3f}")

    print("\nOverall Feature Importance:")
    print(importance_df.round(4).to_string(index=False))
    print("\nSaved results/temporal_attribution.csv and results/feature_importance.csv")
