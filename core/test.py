import joblib
import sklearn

# Load the model
model = joblib.load('../model/decision_tree.pkl')

# Modify the necessary parts of the model
# For instance, if the issue is with references to scikit.tree.tree, you might need to update them manually

# Save the model again
joblib.dump(model, '../model/decision_tree_updated.pkl')
