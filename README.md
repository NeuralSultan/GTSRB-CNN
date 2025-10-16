

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
  <a class="button" href="https://www.linkedin.com/in/fares-sultan-2ba92b324/" target="_blank">Connect on LinkedIn</a>
</section>

<footer>
  <p>Developed by <strong>Fares Sultan</strong> — as part of the <strong>Elevvo Internship</strong></p>
</footer>

</body>
</html>
"""

