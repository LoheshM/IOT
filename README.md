# IOT: machine learning for secure IoT networks, plus a SARIMA forecasting notebook

**A short review of how machine learning can secure IoT networks, and a separate notebook that forecasts a monthly time series with SARIMA.**

The repo holds two pieces of work. [IOT.pdf](IOT.pdf) is a 3-page review paper, *A Review on Machine Learning Techniques for Secure IoT Networks*, written in LaTeX. [iot_timeseries.ipynb](iot_timeseries.ipynb) is a forecasting exercise on [ts_data.csv](ts_data.csv). It does not use IoT or security data.

> Paper dated 22 January 2023 · notebook added April 2023 · Python 3.9, statsmodels

## The problem

IoT networks link many low-power devices that send data to the cloud. They are hard to secure:
- Devices vary widely and are deployed in large numbers.
- They have little computing power for traditional security tools.
- They are exposed to physical access, eavesdropping on wireless links and unreliable connections.

The paper argues that machine learning can help. A model can learn what normal IoT behaviour looks like from the traffic devices already generate, and then flag anomalies, intrusions and attacks.

## Part 1: the review paper

[IOT.pdf](IOT.pdf), by Lohesh M. The LaTeX source is in [IOT.sty](IOT.sty).

| Section | What it covers |
|---|---|
| **1 Introduction** | IoT as a network of low-power devices, the growth of IoT data, why traditional security falls short, and where ML fits. |
| **2 Motivation** | Threats to IoT networks (active and passive). The four properties to protect: confidentiality, authorisation, authentication and integrity, with medical wearables and implants as an example. How ML can profile normal behaviour and detect authentication issues, DDoS attacks and intrusion. |
| **3.1 ML algorithms** | Supervised: SVM, Bayesian / Naive Bayes, k-NN, ANN, ensemble learning. Unsupervised: PCA, k-means. Semi-supervised learning. Reinforcement learning, for example anti-jamming for autonomous radios. Each comes with its typical security use. |
| **3.2 Applications** | Supervised ML for malware analysis, intrusion and attack detection. Unsupervised ML for dimensionality reduction, clustering and anomaly detection. RL for decision-making and against jamming in tactical mobile networks. |
| **4 Limitations** | Too little security-related training data, and privacy problems with confidential data. The paper proposes a crowd-sourcing platform for datasets kept up to date with new attack patterns, and filtering noisy low-level data before real-time training. |
| **5 Conclusion** | Summary of the threats, the ML techniques that answer them, and their limits. |

## Part 2: the time-series notebook

### Dataset

[ts_data.csv](ts_data.csv) has 476 monthly rows with two columns, `Month` and `Production`, from January 1956 to August 1995. The file does not say what is being produced or where the data comes from.

### Method

```
ts_data.csv ─► parse Month as a date, monthly index (freq = 'MS')
             ├─► line plot of the full series
             ├─► seasonal_decompose (additive) ─► trend / seasonal / residual plots
             └─► train = first 464 months (1956-01 … 1994-08)
                 test  = last 12 months  (1994-09 … 1995-08)
                        │
                 SARIMAX(order = (2,1,1), seasonal_order = (4,0,3,12)).fit()
                        │
                 predict the 12 test months ─► plot vs actual, RMSE computed
```

**Notable design decisions**
- **Additive decomposition first.** The decomposition plot shows the trend and a 12-month seasonal pattern before any model is fitted.
- **Seasonal period of 12.** Monthly data with a yearly cycle. One order of non-seasonal differencing (`d = 1`) removes the trend.
- **Hold out the last year.** The model is judged on 12 months it has never seen.

### Results

All values come from the saved outputs of [iot_timeseries.ipynb](iot_timeseries.ipynb). Actual values are from [ts_data.csv](ts_data.csv).

**Model fit** (statsmodels summary, 464 training observations):

| Metric | Value |
|---|---|
| Model | SARIMAX(2, 1, 1)x(4, 0, [1, 2, 3], 12) |
| Log likelihood | -1707.996 |
| AIC | 3437.991 |
| BIC | 3483.506 |
| HQIC | 3455.909 |
| Ljung-Box (L1) Q / p-value | 0.04 / 0.83 |

**12-month forecast vs actual** (error = forecast − actual):

| Month | Actual | Forecast | Error |
|---|---|---|---|
| 1994-09 | 143 | 133.98 | -9.02 |
| 1994-10 | 160 | 157.71 | -2.29 |
| 1994-11 | 190 | 181.88 | -8.12 |
| 1994-12 | 182 | 183.54 | +1.54 |
| 1995-01 | 138 | 144.87 | +6.87 |
| 1995-02 | 136 | 136.87 | +0.87 |
| 1995-03 | 152 | 151.19 | -0.81 |
| 1995-04 | 127 | 133.16 | +6.16 |
| 1995-05 | 151 | 137.96 | -13.04 |
| 1995-06 | 130 | 120.55 | -9.45 |
| 1995-07 | 119 | 128.39 | +9.39 |
| 1995-08 | 153 | 138.85 | -14.15 |

The forecast follows the seasonal shape of the year, including the November–December peak. The largest misses are in May and August 1995. The notebook computes RMSE and MSE but does not print them.

## Repository layout

```
IOT/
├── IOT.pdf               The review paper (3 pages, compiled with pdfTeX)
├── IOT.sty               LaTeX source of the paper (a full document, despite the .sty extension)
├── iot_timeseries.ipynb  SARIMA decomposition and 12-month forecast
├── ts_data.csv           Monthly series, 1956-01 to 1995-08 (Month, Production)
└── README.md
```

## Quick start

**Rebuild the paper.** `IOT.sty` is a complete LaTeX document, so copy it to a `.tex` file first:

```bash
cp IOT.sty IOT.tex
pdflatex IOT.tex
```

**Run the notebook:**

```bash
pip install numpy pandas statsmodels scikit-learn matplotlib jupyter
jupyter notebook iot_timeseries.ipynb
```

The notebook reads the CSV from a hard-coded Windows path. Change that cell to `pd.read_csv("ts_data.csv")`.

## Limitations

- **The notebook is not about IoT security.** It is a general SARIMA exercise on a production series. It does not use network traffic or any security data.
- **The paper is a short overview.** It has no references or citations. Sections 3.1 and 3.2 open with "…" placeholders. The conclusion mentions a comparison of techniques with their advantages, disadvantages and deployment, but the 3-page PDF does not include that comparison.
- **No experiments in the paper.** It describes algorithms and their uses but does not test any of them.
- **SARIMA orders chosen by hand.** `plot_acf` and `plot_pacf` are imported but never used, and no order search is run.
- **Small notebook bugs.** The CSV path is hard-coded. `plt.xlabel=(...)` and `plt.ylabel=(...)` overwrite the functions instead of setting axis labels. RMSE is never printed.
- **Unknown data source.** `ts_data.csv` has no description or source.

## Licence

MIT. See [LICENSE](LICENSE). The MIT licence covers the notebook and LaTeX source. The paper text is by Lohesh M.
