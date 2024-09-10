# **Lab Order Entropy and Ergodicity Analysis**

This project explores the **entropy** and **ergodicity** of laboratory order times across different hospital units and hospitals. The main goal is to show that hospital units exhibit **ergodic properties**, and **entropy** can serve as a marker of **data drift** and **label leakage**. The project analyzes how lab order behavior evolves over time and provides tools to visualize the patterns through entropy and surprisal calculations.

## **Overview**

1. **Entropy**: Measures the unpredictability or randomness in lab order times. This can be used to detect patterns that may indicate **data drift** or **label leakage** in hospital data.
  
2. **Ergodicity**: A property that, when applied to hospital units, suggests that the long-term average behavior of lab orders in a unit is similar to the behavior across different time periods. We use **ergodicity** as a conceptual framework for analyzing how consistent hospital units are in their operations over time.

3. **Data Drift and Label Leakage**: By calculating **entropy** over time, we can detect shifts in lab ordering behavior that may suggest changes in how hospitals function or in how data is labeled. These shifts could indicate potential issues with the integrity of machine learning models built on the data.

## **Scripts**

### 1. `shannon_entropy_by_lab.py`

This script calculates the **Shannon entropy** for lab order times across various hospital units and hospitals. It allows for a detailed analysis of randomness in lab order behavior, which can be used to monitor consistency and data integrity.

#### **Key Features**:
- Calculates **entropy** for lab order times.
- Compares **entropy** across different hospital units.
- Tracks **entropy** over time to detect potential **data drift**.

### 2. `create_surprisal_heatmaps.py`

This script generates heatmaps based on the calculated **surprisal** (a measure of how surprising or rare an event is). The heatmaps provide a visual representation of how **entropy** changes across different hospital units and over time.

#### **Key Features**:
- Visualizes **surprisal** across different hospital units.
- Generates heatmaps to show patterns in lab order times.
- Supports temporal analysis to track changes over time.

## **How to Run**

1. **Install Dependencies**  
   Ensure you have the required Python packages installed. You can install the necessary dependencies using the following command:

   ```bash
   pip install -r requirements.txt
   ```

2. **Run Entropy Calculations**:
  To calculate the Shannon entropy for lab order times, run the following command:

    ```bash
    python shannon_entropy_by_lab.py
    ```

3. **Generate Surprisal Heatmaps**:
  To create heatmaps based on surprisal values, execute the following command:

    ```bash
    python create_surprisal_heatmaps.py
    ```

## **Applications**
- **Data Drift Detection**: Use entropy to detect when hospital units change their lab ordering behavior, which may indicate shifts in operational or clinical practices.
- **Label Leakage Monitoring**: Track entropy as a marker of label leakage in predictive models using hospital data.
- **Consistency Analysis**: Evaluate the ergodicity of hospital units to assess how consistent they are over time in their operations.
Conclusion

This project provides a quantitative framework for analyzing the consistency and predictability of lab order times across hospital units. By leveraging entropy and ergodicity, it offers a novel approach to monitoring hospital operations and ensuring the reliability of predictive models in healthcare settings.
   
