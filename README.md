
# 📦 E-Commerce Shipping Delay Prediction & Optimization Pipeline

## 📌 Project Overview
This repository contains an end-to-end machine learning workflow designed to analyze e-commerce supply chain data and predict logistics delivery outcomes (`Reached.on.Time_Y.N`). Delayed shipments incur massive operational overhead, compromise customer retention, and disrupt fulfillment schedules. This project builds, refines, and evaluates predictive architectures to serve as a reliable tool for preemptive supply chain risk assessment.

The final pipeline contrasts an optimized **Random Forest Classifier** against an advanced **XGBoost (Extreme Gradient Boosting)** framework, prioritizing high-precision validation to make the model operationally actionable for business deployment.

---

## 🛠️ Technical Workflow & Data Engineering

### 1. Preprocessing & Data Integrity
* **Categorical Encoding:** High-cardinality structural features (`Warehouse_block`, `Mode_of_Shipment`, and `Product_importance`) were dynamically encoded into numerical formats compatible with tree-based ensembles.
* **Leakage Prevention & Target Isolation:** Addressed fundamental pipeline bugs by isolating the feature space ($X$) from the target vector ($y$) explicitly by named column reference rather than relative index slicing. This mathematically guaranteed zero target leakage during transformation phases.
* **Data Splitting:** Implemented a clean **80% Training / 20% Testing split** to maintain an entirely unbiased validation holdout.
* **Feature Scaling:** Applied a `StandardScaler` to the continuous variables. Crucially, the scaling parameters were fitted strictly on the training partition and subsequently applied as a static transformation to the test partition, eliminating downstream data leakage.

### 2. Model Optimization & Benchmarking
To identify the most robust deployment candidate, we implemented two separate machine learning frameworks:

* **Baseline Random Forest:** Initial training on default parameters established a raw baseline classification accuracy of **66.05%**.
* **Hyperparameter Optimization (`GridSearchCV`):** Executed a 3-fold cross-validation grid search targeting structural constraints (`max_depth`, `min_samples_split`, `n_estimators`, and `criterion`). Restricting tree depth to control overfitting successfully elevated the optimized Random Forest accuracy to **68.86%**.
* **XGBoost (Extreme Gradient Boosting):** Deployed a sequential gradient boosting framework to minimize residual training errors iteratively. XGBoost yielded a stable overall accuracy of **68.05%** while offering exceptional class-specific reliability.

---

## 📊 Evaluation & Strategic Selection

### 🏆 Deployed Model: XGBoost
While the hyperparameter-tuned Random Forest recorded a marginally higher absolute accuracy score ($68.86\%$ vs $68.05\%$), **XGBoost was selected as the final production model due to its superior Precision profile.**

* **The Precision Breakthrough:** XGBoost achieved an outstanding **85% Precision rate for Delayed Shipments (`1`)**.
* **Operational Impact:** In enterprise logistics, false positives (wrongly flagging an on-time shipment as delayed) incur high financial costs due to unnecessary emergency shipping expedites or proactive client notifications. With an 85% precision threshold, when the model flags a potential logistical bottleneck, management can execute corrective protocols with high statistical confidence.

### 💡 Core Analytical Takeaways (Feature Importance)
Utilizing internal model feature importance calculations, the pipeline exposed the primary operational drivers behind delivery performance:
1. **Product Weight (`Weight_in_gms`)** and **Discounts Offered (`Discount_offered`)** represent the dominant mathematical features determining shipping delays.
2. Structural variables such as `Warehouse_block` and `Mode_of_Shipment` ranked at the absolute bottom of feature significance. 
3. **Conclusion:** Delivery delays are not geographic or transit-dependent bottlenecks. Instead, systemic latency is directly tied to the product physical profile—specifically, heavy bulk package handling and the fulfillment friction caused by high-volume promotional discount ordering.

---

## 💻 Technical Stack
* **Language:** Python 3
* **Libraries:** Pandas, NumPy, Scikit-Learn, XGBoost, Matplotlib, Seaborn
* **Environment:** Google Colab / Jupyter Notebooks
