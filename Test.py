import pandas as pd

# Manually extracted values from the image for 9 rows

data = {
    "Piston Roundness": [0.003, 0.000996, None, None, None, None, None, 0.000705, None],
    "Piston Diameter": [18.47, 18.47, 18.47, 18.47, 18.474, 18.474, 18.474, 18.474, 18.474],
    "Cylinder Diameter": [18.5065, 18.5088, 18.5067, 18.5086, 18.5089, 18.5061, 18.5087, 18.5085, 18.5102],
    "Cylinder Position": [0.0189, 0.0207, 0.0091, 0.0023, 0.0064, 0.0028, 0.008, 0.0197, 0.0199],
    "Cylinder Coaxiality": [0.288, 0.0664, 0.0710, 0.0660, 0.0520, 0.0696, 0.0594, 0.0424, 0.0268],
    "Cylinder Roundness": [0.0071, 0.0080, 0.0073, 0.0086, 0.0073, 0.0071, 0.0079, 0.0089, 0.0076]
}

df = pd.DataFrame(data)

# Save to CSV
csv_path = "piston_cylinder_combined_data.csv"
df.to_csv(csv_path, index=False)

csv_path
