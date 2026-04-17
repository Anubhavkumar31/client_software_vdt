# import pandas as pd
#
# def run_clustering(df, distance_threshold=100):
#     """
#     Simple distance-based clustering for metal loss defects
#     """
#
#     if df is None or df.empty:
#         return []
#
#     # Column check (safe)
#     if "Abs. Distance (m)" not in df.columns:
#         raise ValueError("Column 'Abs. Distance (m)' not found in pipe tally")
#
#     # Sort by distance
#     df = df.sort_values("Abs. Distance (m)").reset_index(drop=True)
#
#     clusters = []
#     current_cluster = []
#
#     for _, row in df.iterrows():
#         if not current_cluster:
#             current_cluster.append(row)
#         else:
#             last_row = current_cluster[-1]
#             if abs(row["Abs. Distance (m)"] - last_row["Abs. Distance (m)"]) <= distance_threshold:
#                 current_cluster.append(row)
#             else:
#                 clusters.append(current_cluster)
#                 current_cluster = [row]
#
#     if current_cluster:
#         clusters.append(current_cluster)
#
#     return clusters



def run_clustering(df, distance_threshold=1.0):
    """
    Groups defects into clusters based on Abs. Distance (m)

    distance_threshold:
        Maximum distance (in meters) between defects
        to be considered in the same cluster
    """

    if df is None or df.empty:
        return []

    # 🔹 Ensure required column exists
    if "Abs. Distance (m)" not in df.columns:
        raise ValueError("Required column 'Abs. Distance (m)' not found in pipe tally")

    # 🔹 Convert distance column to numeric (safe)
    df = df.copy()
    df["Abs. Distance (m)"] = (
        df["Abs. Distance (m)"]
        .astype(str)
        .str.strip()
        .replace("", None)
    )

    df["Abs. Distance (m)"] = df["Abs. Distance (m)"].astype(float)

    # 🔹 Drop rows with invalid distance
    df = df.dropna(subset=["Abs. Distance (m)"])

    # 🔹 Sort by distance (IMPORTANT)
    df = df.sort_values("Abs. Distance (m)").reset_index(drop=True)

    clusters = []
    current_cluster = []

    for _, row in df.iterrows():
        row_dict = row.to_dict()

        if not current_cluster:
            current_cluster.append(row_dict)
            continue

        last_distance = current_cluster[-1]["Abs. Distance (m)"]
        current_distance = row_dict["Abs. Distance (m)"]

        # 🔹 Same cluster if distance difference <= threshold
        if abs(current_distance - last_distance) <= distance_threshold:
            current_cluster.append(row_dict)
        else:
            clusters.append(current_cluster)
            current_cluster = [row_dict]

    # 🔹 Append final cluster
    if current_cluster:
        clusters.append(current_cluster)

    return clusters








# def build_cluster_rows(clusters):
#     """
#     Convert clusters into table rows
#     """
#     rows = []
#
#     for idx, cluster in enumerate(clusters, start=1):
#         distances = [r["Abs. Distance (m)"] for r in cluster]
#
#         rows.append({
#             "Feature Type": "CLUSTER",
#             "Cluster ID": idx,
#             "Abs. Distance (m)": sum(distances) / len(distances),
#             "No. of Defects": len(cluster)
#         })
#
#     return rows


def build_cluster_rows(clusters):
    rows = []

    for idx, cluster in enumerate(clusters, start=1):
        distances = []

        for r in cluster:
            val = r.get("Abs. Distance (m)")

            if val is None:
                continue

            try:
                val = float(val)
                distances.append(val)
            except (ValueError, TypeError):
                continue

        # Skip invalid cluster
        if not distances:
            continue

        rows.append({
            "Feature Type": "Metal Loss",
            "Cluster ID": idx,

            # ✅ MANUAL-COMPLIANT representative distance
            "Abs. Distance (m)": round(min(distances), 3),

            "No. of Defects": len(cluster)
        })

    return rows


