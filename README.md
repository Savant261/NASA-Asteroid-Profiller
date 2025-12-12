# ☄️ NASA Asteroid Profiler

A machine learning project that uses a 3-stage AI pipeline to profile, measure, and assess the potential hazard of Near-Earth Objects (NEOs). This system provides an interactive dashboard built with Streamlit to visualize data and make live predictions.

---

## 🎯 Features

*   **3-Stage AI Pipeline:**
    1.  **Clustering (The Sorter):** Classifies asteroids into orbital families (Amor, Apollo, Aten) using a K-Means clustering model.
    2.  **Regression (The Sizer):** Estimates the diameter of an asteroid using a Random Forest Regressor.
    3.  **Classification (The Guard):** Determines if an asteroid is potentially hazardous using a Random Forest Classifier.
*   **Interactive Dashboard:** A user-friendly web interface built with Streamlit to input orbital parameters and receive instant threat analysis.
*   **Data Visualization:** Includes interactive 3D and 2D plots to visualize the asteroid's orbital family and its position in the hazard risk zone.
*   **Pre-trained Models:** The project comes with pre-trained models for immediate use.

---

## 🔧 Tech Stack

This project is built with the following technologies:

*   **Python:** The core programming language.
*   **Pandas:** For data manipulation and analysis.
*   **Scikit-learn:** For machine learning modeling and data preprocessing.
*   **Streamlit:** To create and serve the interactive web dashboard.
*   **Plotly:** For creating interactive visualizations.
*   **Pickle:** For saving and loading the trained machine learning models.

---

## 🚀 Getting Started

Follow these instructions to get a copy of the project up and running on your local machine.

### Prerequisites

You need to have Python 3.8 (or newer) and pip installed on your system.

### Installation

1.  **Clone the repository:**

    ```bash
    git clone https://github.com/your-username/nasa-asteroid-profiler.git
    cd nasa-asteroid-profiler
    ```

2.  **Create a virtual environment** (recommended):

    ```bash
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate

    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    ```

3.  **Install the required dependencies:**

   Run the following command to install the packages:

    ```bash
    pip install -r requirements.txt
    ```

---

## 🏃‍♂️ How to Run

Once you have installed the dependencies, you can run the Streamlit app locally:

```bash
streamlit run app.py
```

Open your web browser and go to **`http://localhost:8501`** to see the application in action!

---

## 📂 Project Structure

Here is an overview of the key files in this project:

```
.
├── 📜 app.py                     # The main Python script for the Streamlit app
├── 📜 data_aggregation.py        # Script for data collection and preprocessing
├── 📜 clustering_model.py        # Script to train the K-Means clustering model
├── 📜 size_regression.py         # Script to train the diameter regression model
├── 📜 asteroid_classification.py # Script to train the hazard classification model
├── 📜 visualizations.py          # Script for generating visualizations
├── 🧠 kmeans_model.pkl           # The pre-trained K-Means clustering model
├── 🧠 diameter_regressor.pkl     # The pre-trained diameter regression model
├── 🧠 hazard_model.pkl           # The pre-trained hazard classification model
├── 🧠 scaler.pkl                 # The pre-trained data scaler
├── 📄 nasa_asteroid_clustered.csv  # Dataset with cluster labels
├── 📄 nasa_asteroid_predicted.csv# Dataset with predicted diameters
└── README.md                    # You are here!
```

---

## 📈 The Models

The project uses a pipeline of three machine learning models:

1.  **K-Means Clustering:**
    *   **Purpose:** To group asteroids into orbital families based on their `semi_major_axis`, `eccentricity`, and `inclination`.
    *   **Clusters:** 3 (Amor, Apollo, Aten).

2.  **Random Forest Regressor:**
    *   **Purpose:** To predict the `diameter` of an asteroid.
    *   **Features:** `magnitude_h`, `semi_major_axis`, `eccentricity`, `inclination`.

3.  **Random Forest Classifier:**
    *   **Purpose:** To predict if an asteroid is a `Potentially Hazardous Asteroid (PHA)`.
    *   **Features:** `moid_au`, `diameter`, `eccentricity`, `semi_major_axis`, `inclination`, `cluster_label`.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions for improving the models or the app, please feel free to fork the repo and create a pull request.

1.  Fork the Project
2.  Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3.  Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4.  Push to the Branch (`git push origin feature/AmazingFeature`)
5.  Open a Pull Request
