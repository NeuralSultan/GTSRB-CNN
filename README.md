# Creating an HTML version of the README and saving it to /mnt/data/README.html
html_content = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<meta name="viewport" content="width=device-width,initial-scale=1"/>
<title>GTSRB Traffic Sign Classification using CNN - README</title>
<style>
  body { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial; line-height:1.6; color:#111; padding:24px; max-width:900px; margin:0 auto; background:#f7f9fb; }
  header { background:linear-gradient(90deg,#0f172a,#0ea5a0); color:white; padding:24px; border-radius:12px; margin-bottom:20px; box-shadow:0 6px 18px rgba(12,12,15,0.08); }
  h1 { margin:0; font-size:1.6rem; }
  .badge { display:inline-block; margin-top:8px; padding:6px 10px; background:rgba(255,255,255,0.12); border-radius:999px; font-size:0.9rem; }
  section { background:white; padding:18px; border-radius:10px; box-shadow:0 4px 12px rgba(12,12,15,0.04); margin-bottom:16px; }
  pre { background:#0b0f1a; color:#e6eef8; padding:12px; border-radius:8px; overflow:auto; }
  code { background:#eef2f7; padding:2px 6px; border-radius:6px; font-family:monospace; }
  table { border-collapse:collapse; width:100%; }
  table td, table th { border:1px solid #eef0f3; padding:8px; text-align:left; }
  .muted { color:#556; font-size:0.95rem; }
  .center { text-align:center; }
  a.button { display:inline-block; margin-top:8px; padding:8px 12px; background:#0ea5a0; color:white; border-radius:8px; text-decoration:none; }
  footer { text-align:center; color:#666; font-size:0.95rem; margin-top:18px; }
</style>
</head>
<body>

<header>
  <h1>🚦 GTSRB Traffic Sign Classification using CNN</h1>
  <div class="badge">Developed during internship at <strong>Elevvo</strong></div>
</header>

<section>
  <h2>About</h2>
  <p>This project focuses on classifying <strong>German Traffic Signs</strong> from the <em>GTSRB (German Traffic Sign Recognition Benchmark)</em> dataset using a <strong>Convolutional Neural Network (CNN)</strong> built with TensorFlow and Keras.</p>
</section>

<section>
  <h2>Project Structure</h2>
  <pre>
GTSRB-CNN/
│
├── GTSRB.py              # Main training and evaluation script
├── gtsrb_cnn.keras       # Saved model (after training)
├── README.md             # Project documentation
└── data/                 # Dataset folder (not included due to size)
  </pre>
</section>

<section>
  <h2>Model Architecture</h2>
  <p>The model is a <strong>custom CNN</strong> built from scratch using TensorFlow/Keras.</p>
  <p><strong>Layers used:</strong></p>
  <ul>
    <li>Convolutional + ReLU + Batch Normalization</li>
    <li>MaxPooling</li>
    <li>Global Average Pooling</li>
    <li>Dense + Dropout</li>
    <li>Output layer (Softmax for 43 classes)</li>
  </ul>
  <p><strong>Key features:</strong></p>
  <ul>
    <li>Data augmentation (rotation, zoom, contrast, translation)</li>
    <li>EarlyStopping &amp; ModelCheckpoint callbacks</li>
    <li>Stratified train/validation split</li>
  </ul>
</section>

<section>
  <h2>Dataset</h2>
  <ul>
    <li><strong>Dataset:</strong> GTSRB (German Traffic Sign Recognition Benchmark)</li>
    <li><strong>Classes:</strong> 43</li>
    <li><strong>Source:</strong> Replace the placeholder link in the original README with the correct dataset URL if needed.</li>
    <li><strong>Preprocessing:</strong> Image resizing to 64×64 and normalization (<code>Rescaling(1./255)</code>).</li>
  </ul>
</section>

<section>
  <h2>Training Configuration</h2>
  <table>
    <tr><th>Parameter</th><th>Value</th></tr>
    <tr><td>Image Size</td><td>64×64</td></tr>
    <tr><td>Batch Size</td><td>64</td></tr>
    <tr><td>Epochs</td><td>16</td></tr>
    <tr><td>Optimizer</td><td>Adam</td></tr>
    <tr><td>Loss</td><td>Sparse Categorical Crossentropy</td></tr>
  </table>
</section>

<section>
  <h2>Results</h2>
  <table>
    <tr><th>Metric</th><th>Value</th></tr>
    <tr><td><strong>Test Accuracy</strong></td><td>✅ <strong>97.47%</strong></td></tr>
  </table>
</section>

<section>
  <h2>Visualizations</h2>
  <p>The project produces the following visualizations during/after training:</p>
  <ul>
    <li>Confusion Matrix</li>
    <li>Classification Report</li>
    <li>Training vs Validation Accuracy &amp; Loss Curves</li>
    <li>Random prediction visualizations (correct vs incorrect samples)</li>
  </ul>
</section>

<section>
  <h2>Requirements</h2>
  <p>Install dependencies using:</p>
  <pre>pip install tensorflow numpy pandas matplotlib seaborn scikit-learn opencv-python</pre>
</section>

<section>
  <h2>How to Run</h2>
  <ol>
    <li>Clone this repository:
      <pre>git clone https://github.com/USERNAME/GTSRB-CNN.git
cd GTSRB-CNN</pre>
    </li>
    <li>Place the dataset inside the folder:
      <pre>data/Train.csv
data/Test.csv</pre>
      (and all related images under the same directory)
    </li>
    <li>Run the training script:
      <pre>python GTSRB.py</pre>
    </li>
  </ol>
</section>

<section>
  <h2>Conclusion</h2>
  <p>This project successfully classifies German traffic signs using a CNN model with a <strong>test accuracy of 97.47%</strong>. It demonstrates how deep learning can effectively be applied to image classification and road safety automation.</p>
</section>

<section class="center">
  <a class="button" href="https://www.linkedin.com" target="_blank">Connect on LinkedIn</a>
</section>

<footer>
  <p>Developed by <strong>Fares Sultan</strong> — as part of the <strong>Elevvo Internship</strong></p>
</footer>

</body>
</html>
"""

file_path = "/mnt/data/README.html"
with open(file_path, "w", encoding="utf-8") as f:
    f.write(html_content)

file_path

