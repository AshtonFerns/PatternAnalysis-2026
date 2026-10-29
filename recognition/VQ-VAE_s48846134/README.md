# Generating Realistic Prostate MRI Slices with VQ-VAE and PixelCNN

## 1. Problem and Algorithm


## 2. Feasibility Report

### User Need, Scope & Acceptance Criteria

The intended user is a radiotherapy research group that wants to share or enlarge prostate MRI training data without exposing patient data. The prototype must show that the generative model can produce unique and diverse but faithful 2D slices on consumer hardware.

- **AC1:** Reconstruction fidelity on unseen test patients must have mean SSIM > 0.75 and 95% of slices must have SSIM > 0.6 (SSIM is the difference in brightness, contrast and structure).
- **AC2:** There must be no memorisation, so the median RMSE (pixel difference to the most similar training slice) for generated slices must not be more than 1% below that of unseen patient images, and none should have an RMSE lower than half the unseen patient median.
- **AC3:** Must be trainable on a basic consumer GPU with 8GB of VRAM (RX 6600 XT) in under 5 hours, and generating 1 slice must take less than 1 second.

### Model Choice & Course Concepts

A VQ-VAE network (hard difficulty) will be trained to encode and decode 2D slices into discrete codebooks, with the goal of accurate image reproduction. This will be paired with a PixelCNN prior network, which will be trained to generate realistic codebooks that decode to new 2D slices. VQ-VAE builds upon the VAE architecture used in the last lab by adding a quantisation layer to the middle of the network. Without this, VAEs tend to produce blurry, unconfident outputs, and encounter posterior collapse. So, with PixelCNN, this architecture should produce realistic, sharp images useable by researchers. This will be compared against a baseline VAE network with points sampled randomly from its latent space. The training set contains 11460 slices from 31 patients, the validation set contains 660 slices from 4 patients, and the test set contains 540 slices from 3 patients. There is no patient leakage.

### Preliminary Feasibility Evidence

The original 2D slices have a height and width of 256 x 128, with the exception of 60 slices from one scan (case_019_week_1), which were 256 x 144. These slices were dropped from the dataset. Images from the dataset can have different brightness levels and noise, so the images were normalised by scaling the brightness of all pixels so that the top 0.5% are given a brightness of 1. The background pixels were confirmed to have a brightness of 0.

Three VQ-VAE networks were trained for 4 epochs (took around 10 minutes each) and tested. The network with 2 stages, which shrinks the grid size by 4, had an SSIM of 0.940. The network with 3 stages (8x shrink) had an SSIM of 0.831, and the network with 4 stages (16x shrink) had an SSIM of 0.676. As too large of a codebook would slow down the PixelCNN training and sampling, the 3 stage network with a reasonably high reconstruction score was chosen as the base for the final VQ-VAE network.

![Downsample testing reconstructions](figures/downsample_testing_reconstructions.png)

### Risks, Budget & Fallback

**Risks:** Low number of test patients could result in inaccurate results. Codebook collapse could happen, which is when a network doesn’t use every codebook entry properly. This can be tracked with perplexity and managed using EMA.

**Budget:** <5 hours of training on the RX 6600 XT with ROCm PyTorch, and <1 second to generate each slice.

**Fallback:** The 2 stage network will be used instead if the solution doesn’t perform well enough, and the 4 stage network will be used if inference isn’t fast enough.


## 3. Dependencies and Usage


## 4. Usage


## 5. Data, Pre-processing and Splits


## 6. Training and Results


## 7. Open research dilemma


## 8. AI Usage Disclosure

The Cursor IDE (with Auto selected) was used to show me how to write the code that does what I want, and troubleshoot issues whenever I hit roadblocks. 
I copied and pasted parts of this code, but wrote most of it myself, and ensured that I understood what every part of the code was doing before each commit. 
