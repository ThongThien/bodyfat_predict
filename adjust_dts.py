import numpy as np
import pandas as pd


def create_final_dataset(input_path, output_path):
    """Create the final dataset by adding derived anthropometric features."""
    print("--- Starting feature engineering process ---")

    try:
        df = pd.read_csv(input_path)
    except Exception:
        df = pd.read_excel(input_path)

    print(f"Loaded rows: {len(df)}")

    # Create derived features from body measurements.
    df["W_per_A"] = (df["Abdomen"] ** 2) / df["Weight"]
    df["WHR"] = df["Abdomen"] / df["Hip"]
    df["WtHR"] = df["Abdomen"] / df["Height"]

    final_columns = [
        "BodyFat",
        "Weight",
        "Chest",
        "Abdomen",
        "Hip",
        "W_per_A",
        "WtHR",
        "WHR",
        "Height",
        "Age",
    ]

    final_columns = [col for col in final_columns if col in df.columns]
    final_dataset = df[final_columns]

    print("\n[LOG] Preview of the final dataset:")
    print(final_dataset[["BodyFat", "Weight", "Abdomen", "W_per_A"]].head())

    final_dataset.to_csv(output_path, index=False)

    print("-" * 30)
    print(f"[SUCCESS] File created: {output_path}")
    print(f"Total rows: {len(final_dataset)}")
    print(f"Selected columns: {final_dataset.columns.tolist()}")


if __name__ == "__main__":
    create_final_dataset("bodyfat_final_v1.csv", "bodyfat_final_v2.csv")