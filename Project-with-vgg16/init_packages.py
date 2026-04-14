import os

# Create __init__.py files
for d in ['src', 'utils', 'ui']:
    with open(os.path.join(d, "__init__.py"), "w") as f:
        pass

print("Initialized Python packages.")
