<h1>RNN Stock Trend Analyzer</h1>
<hr><p>A Recurrent Neural Net designed to predict trend values of stocks.</p><h2>General Information</h2>
<hr><ul>
<li>Implements an LSTM model in PyTorch to predict future trends in the price of a given stock.</li>
</ul><ul>
<li>Allows an investor the ability to see patterns and potential trends in a stock that other machine learning models cannot catch.</li>
</ul><ul>
<li>Purpose is to recreate the research done in the following paper on machine learning models for stock prediction</li>
</ul>
<p>https://ieeexplore.ieee.org/abstract/document/9165760</p><h2>Technologies Used</h2>
<hr><ul>
<li>Python</li>
</ul><ul>
<li>Yahoo Finance API</li>
</ul><ul>
<li>PyTorch</li>
</ul><ul>
<li>Pandas</li>
</ul><h2>Setup</h2>
<hr><p>Requires Python 3.11.9<br>
Requires CUDA Toolkit (12.4 or 12.8)</p><h5>Steps</h5><ul>
<li>git clone https://github.com/Warping/RNN-Stocker.git</li>
</ul><ul>
<li>cd RNN-Stocker</li>
</ul><ul>
<li>pip install -r requirements.txt</li>
</ul><ul>
<li>pip install --pre torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu128</li>
</ul><ul>
<li>python.exe .\lstm3.py</li>
</ul><h2>Usage</h2>
<hr><p>Can modify hyperparameters, early stopping rules, stock data preprocessing</p><h2>Project Status</h2>
<hr><p>Currently in progress: ETA - May 1st 2025</p><h2>Improvements</h2>
<hr><ul>
<li>Better Validation Set Selection</li>
</ul><ul>
<li>Fix overfitting issues DONE!</li>
</ul><ul>
<li>Fix underfitting on binarized data</li>
</ul><h2>Features that can be added</h2>
<hr><ul>
<li>Add trendline prediction visualizations DONE!</li>
</ul><ul>
<li>Add real-time model training on current data</li>
</ul><ul>
<li>Add more input features for sentiment analysis and economic growth factors</li>
</ul><h2>Acknowledgement</h2>
<hr><ul>
<li>Special Thanks to Professor Moghadam for inspiration and the motivation to build this</li>
</ul><ul>
<li>References: https://ieeexplore.ieee.org/abstract/document/9165760</li>
</ul>
