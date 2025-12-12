# Machine Learning and Deep Learning Methods for Electricity Day-Ahead Price Forecasting

### Abstract
As socio-economic dependence on electricity prices increases, the ability to anticipate price fluctuations is essential for maintaining system stability. While statistical forecasting methods attempt to address this need, artificial neural networks often fail to capture the innate volatility of electricity prices due to overfitting. Other approaches, including modern machine learning methods and energy system models, often perform better during volatile periods but demonstrate reduced accuracy under typical price fluctuations. To address this challenge, we systematically evaluate the performance of deep learning, machine learning, and data manipulation methodologies across both stable and volatile market conditions. The models we use consider a wide range of external variables, like grid load, cross-border electricity trading volume, and weather, to capture more complex and long-term patterns in prices. Logistical constraints regarding real-time electricity trading–such as prices being set for each hour the next day, all at once on the day before–are carefully considered in our dataset formation. Additionally, techniques like the mirror-logarithmic transformation and target normalization are used for reducing the impact of historical price spikes in model fitting. Ultimately, we compare numerous modern machine learning methods against more complex deep learning models, including a contrast of bias and variance metrics of test set predictions among models.

### Repository Structure
de_lu_epf/

├── data/

├── doc/

├── models/

├── results/

├── src/

├── LICENSE

└── README.md

* data/
Contains all datasets.

* doc/
Jupyter notebooks for research work and thought process.

* models/
Stores trained model artifacts and related metadata.

* results/
Diagrams, plots, tables, and reports for analysis.

* src/
All source code, Python files, and utility scripts.

* LICENSE
License governing the project’s use and distribution.

* README.md
Top-level overview and project description.
