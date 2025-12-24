#%%
import matplotlib.pyplot as plt
import numpy as np
import pynapple as nap

import nemos as nmo

# some helper plotting functions
from nemos import _documentation_utils as doc_plots

# configure pynapple to ignore conversion warning
nap.nap_config.suppress_conversion_warnings = True

# configure plots some
plt.style.use(nmo.styles.plot_style)


# %%
path = nmo.fetch.fetch_data("Mouse32-140822.nwb")

data = nap.load_file(path)

data
# %%
spikes = data["units"]

spikes
# %%
epochs = data["epochs"]
wake_ep = epochs[epochs.tags == "wake"]

angle = data["ry"]
spikes = spikes.getby_category("location")["adn"]

# important! only get spikes that are greater than 1 Hz - take active units only
spikes = spikes.restrict(wake_ep).getby_threshold("rate", 1.0)

# restrict the angle data to the wake epoch
angle = angle.restrict(wake_ep)
# %%
tuning_curves = nap.compute_tuning_curves(
    spikes, features=angle, bins=360, range=(0, 2 * np.pi), feature_names=["angles"]
)

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].plot(tuning_curves[0])
ax[0].set_xlabel("Angle (rad)")
ax[0].set_ylabel("Firing rate (Hz)")
ax[1].plot(tuning_curves[1])
ax[1].set_xlabel("Angle (rad)")
plt.tight_layout()
# %%
fig = doc_plots.plot_head_direction_tuning(
    tuning_curves, spikes, angle, threshold_hz=1, start=8910, end=8960
)
# %%
wake_ep = nap.IntervalSet(
    start=wake_ep.start[0], end=wake_ep.start[0] + 3 * 60
)

bin_size = 0.01
count = spikes.count(bin_size, ep=wake_ep)

pref_ang = tuning_curves.idxmax(dim="angles")

count = nap.TsdFrame(
    t=count.t,
    d=count.values[:, np.argsort(pref_ang)],
)
#%%
# select a neuron's spike count time series
neuron_count = count[:, 3]

# restrict to a smaller time interval
epoch_one_spk = nap.IntervalSet(
    start=count.time_support.start[0], end=count.time_support.start[0] + 1.2
)
plt.figure(figsize=(8, 3.5))
plt.step(
    neuron_count.restrict(epoch_one_spk).t, neuron_count.restrict(epoch_one_spk).d, where="post"
)
plt.title("Spike Count Time Series")
plt.xlabel("Time (sec)")
plt.ylabel("Counts")
plt.tight_layout()
# %%
# set the size of the spike history window in seconds
window_size_sec = 0.8

doc_plots.plot_history_window(neuron_count, epoch_one_spk, window_size_sec);
# %%
# convert the prediction window to bins (by multiplying with the sampling rate)
window_size = int(window_size_sec * neuron_count.rate)

# convolve the counts with the identity matrix.
plt.close("all")
input_feature = nmo.convolve.create_convolutional_predictor(
    np.eye(window_size), neuron_count
)

# print the NaN indices along the time axis
print("NaN indices:\n", np.where(np.isnan(input_feature[:, 0]))[0])
# %%
print(f"Time bins in counts: {neuron_count.shape[0]}")
print(f"Convolution window size in bins: {window_size}")
print(f"Feature shape: {input_feature.shape}")
# %%
suptitle = "Input feature: Count History"
neuron_id = 0
doc_plots.plot_features(input_feature, count.rate, suptitle);
# %%
# construct the train and test epochs
duration = input_feature.time_support.tot_length("s")
start = input_feature.time_support["start"]
end = input_feature.time_support["end"]
first_half = nap.IntervalSet(start, start + duration / 2)
second_half = nap.IntervalSet(start + duration / 2, end)
# %%
# define the GLM object
model = nmo.glm.GLM(solver_name="LBFGS")

# Fit over the training epochs
model.fit(
    input_feature.restrict(first_half),
    neuron_count.restrict(first_half)
)
# %%
plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, np.squeeze(model.coef_), lw=2, label="GLM raw history 1st Half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()
# %%
# fit on the test set

model_second_half = nmo.glm.GLM(solver_name="LBFGS")
model_second_half.fit(
    input_feature.restrict(second_half),
    neuron_count.restrict(second_half)
)

plt.figure()
plt.title("Spike History Weights")
plt.plot(np.arange(window_size) / count.rate, np.squeeze(model.coef_),
         label="GLM raw history 1st Half", lw=2)
plt.plot(np.arange(window_size) / count.rate,  np.squeeze(model_second_half.coef_),
         color="orange", label="GLM raw history 2nd Half", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time From Spike (sec)")
plt.ylabel("Kernel")
plt.legend()
# %%
doc_plots.plot_basis();
# %%
# a basis object can be instantiated in "conv" mode for convolving  the input.
basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=8, window_size=window_size, label="count_history"
)
print(basis)

# `basis.evaluate_on_grid` is a convenience method to view all basis functions
# across their whole domain:
time, basis_kernels = basis.evaluate_on_grid(window_size)

print(basis_kernels.shape)

# time takes equi-spaced values between 0 and 1, we could multiply by the
# duration of our window to scale it to seconds.
time *= window_size_sec
# %%
# compute the least-squares weights
lsq_coef, _, _, _ = np.linalg.lstsq(basis_kernels, np.squeeze(model.coef_), rcond=-1)

# plot the basis and the approximation
doc_plots.plot_weighted_sum_basis(time, model.coef_, basis_kernels, lsq_coef);
# %%
# equivalent to
# `nmo.convolve.create_convolutional_predictor(basis_kernels, neuron_count)`
conv_spk = basis.compute_features(neuron_count)

print(f"Raw count history as feature: {input_feature.shape}")
print(f"Compressed count history as feature: {conv_spk.shape}")

# Visualize the convolution results
epoch_one_spk = nap.IntervalSet(8917.5, 8918.5)
epoch_multi_spk = nap.IntervalSet(8979.2, 8980.2)

doc_plots.plot_convolved_counts(neuron_count, conv_spk, epoch_one_spk, epoch_multi_spk);

# find interval with two spikes to show the accumulation, in a second row
# %%
# use restrict on interval set training
model_basis = nmo.glm.GLM(solver_name="LBFGS")
model_basis.fit(conv_spk.restrict(first_half), neuron_count.restrict(first_half))

# %%
print(model_basis.coef_)
# %%
self_connection = np.matmul(basis_kernels, np.squeeze(model_basis.coef_))

print(self_connection.shape)
# %%
plt.figure()
plt.title("Spike History Weights")
plt.plot(time, np.squeeze(model.coef_), alpha=0.3, label="GLM raw history")
plt.plot(time, self_connection, "--k", label="GLM basis", lw=2)
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")
plt.legend()

# %%
model_basis_second_half = nmo.glm.GLM(solver_name="LBFGS")
model_basis_second_half.fit(conv_spk.restrict(second_half), neuron_count.restrict(second_half))

# compute responses for the 2nd half fit
self_connection_second_half = np.matmul(basis_kernels, np.squeeze(model_basis_second_half.coef_))

plt.figure()
plt.title("Spike History Weights")
plt.plot(time, np.squeeze(model.coef_), "k", alpha=0.3, label="GLM raw history 1st half")
plt.plot(time, np.squeeze(model_second_half.coef_), alpha=0.3, color="orange", label="GLM raw history 2nd half")
plt.plot(time, self_connection, "--k", lw=2, label="GLM basis 1st half")
plt.plot(time, self_connection_second_half, color="orange", lw=2, ls="--", label="GLM basis 2nd half")
plt.axhline(0, color="k", lw=0.5)
plt.xlabel("Time from spike (sec)")
plt.ylabel("Weight")
plt.legend()

# %%
# compare model scores, as expected the training score is better with more parameters
# this may could be over-fitting.
print(f"full history train score: {model.score(input_feature.restrict(first_half), neuron_count.restrict(first_half), score_type='pseudo-r2-Cohen')}")
print(f"basis train score: {model_basis.score(conv_spk.restrict(first_half), neuron_count.restrict(first_half), score_type='pseudo-r2-Cohen')}")
print(f"\nfull history test score: {model.score(input_feature.restrict(second_half), neuron_count.restrict(second_half), score_type='pseudo-r2-Cohen')}")
print(f"basis test score: {model_basis.score(conv_spk.restrict(second_half), neuron_count.restrict(second_half), score_type='pseudo-r2-Cohen')}")

# %%
rate_basis = model_basis.predict(conv_spk) * conv_spk.rate
rate_history = model.predict(input_feature) * conv_spk.rate
ep = nap.IntervalSet(start=8819.4, end=8821)

# plot the rates
doc_plots.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection raw history":rate_history, "Self-connection bsais": rate_basis}
);

# %%
# re-initialize basis
basis = nmo.basis.RaisedCosineLogConv(
    n_basis_funcs=8, window_size=window_size, label="population_history"
)

# convolve all the neurons
convolved_count = basis.compute_features(count)
print(f"Convolved count shape: {convolved_count.shape}")

# %%
model = nmo.glm.PopulationGLM(
    regularizer="Ridge",
    solver_name="LBFGS",
    regularizer_strength=0.1
    ).fit(convolved_count, count)

# %%
predicted_firing_rate = model.predict(convolved_count) * conv_spk.rate
# use pynapple for time axis for all variables plotted for tick labels in imshow
doc_plots.plot_head_direction_tuning_model(tuning_curves, predicted_firing_rate, spikes, angle, threshold_hz=1,
                                          start=8910, end=8960, cmap_label="hsv");
# %%
# mkdocs_gallery_thumbnail_number = 2
fig = doc_plots.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection: raw history": rate_history,
     "Self-connection: bsais": rate_basis,
     "All-to-all: basis": predicted_firing_rate[:, 0]}
)

# %%
# mkdocs_gallery_thumbnail_number = 2
fig = doc_plots.plot_rates_and_smoothed_counts(
    neuron_count,
    {"Self-connection: raw history": rate_history,
     "Self-connection: bsais": rate_basis,
     "All-to-all: basis": predicted_firing_rate[:, 0]}
)

# use split_by_feature to reshape the coefficients
weights_dict = basis.split_by_feature(model.coef_, axis=0)

# get the component weights indexing with the basis label.
weights = weights_dict["population_history"]
print(weights.shape)

responses = np.einsum("jki,tk->ijt", weights, basis_kernels)

# %%
