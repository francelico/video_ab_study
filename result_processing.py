import pandas as pd

df = pd.read_csv("~/Downloads/results.csv")

left = (
    df[["participant_id", "trial_index", "set_name", "metric_a_left", "metric_b_left", "metric_c_left", "metric_d_left"]]
    .rename(columns={
        "metric_a_left": "metric_a",
        "metric_b_left": "metric_b",
        "metric_c_left": "metric_c",
        "metric_d_left": "metric_d",
    })
    .assign(video="A")
)

right = (
    df[["participant_id", "trial_index", "set_name", "metric_a_right", "metric_b_right", "metric_c_right", "metric_d_right"]]
    .rename(columns={
        "metric_a_right": "metric_a",
        "metric_b_right": "metric_b",
        "metric_c_right": "metric_c",
        "metric_d_right": "metric_d",
    })
    .assign(video="B")
)

long = pd.concat([left, right], ignore_index=True)

# Optional: ensure numeric dtype
long[["metric_a", "metric_b", "metric_c", "metric_d"]] = long[["metric_a", "metric_b", "metric_c", "metric_d"]].apply(pd.to_numeric)

print(long.head())
print(long.dtypes)
