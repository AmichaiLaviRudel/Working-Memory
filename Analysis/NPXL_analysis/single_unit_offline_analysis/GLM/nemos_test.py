#%%
import nemos as nmo

# Instantiate the single model
model = nmo.glm.GLM()

import numpy as np
num_samples, num_features = 100, 3

# Generate a design matrix
X = np.random.normal(size=(num_samples, num_features))
# Insert some NaN values into X
X[5:10, 0] = np.nan  # NaN in first feature for samples 5-9
X[15:20, 1] = np.nan  # NaN in second feature for samples 15-19
# generate some counts
spike_counts = np.random.poisson(size=num_samples)

# define fit the model
model = model.fit(X, spike_counts)

# model coefficients shape is (num_features, )
print(f"Model coefficients shape: {model.coef_.shape}")

# model intercept, shape (1,) since there is only one neuron.
print(f"Model intercept shape: {model.intercept_.shape}")

# predict the rate
predicted_rate = model.predict(X)
# firing rate has shape: (num_samples,)
predicted_rate.shape


# compute the log-likelihood of the model
log_likelihood = model.score(X, spike_counts)
# %%
