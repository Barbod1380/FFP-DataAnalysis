# 📄 Pipeline Defect Analysis – Data Dictionary & Column Descriptions

Welcome to the documentation for the **Fitness for Service (FFS)** data-analysis application for oil and gas pipelines. This document explains the structure of the input `.csv` dataset, defining what each column represents and how the information is interpreted.

This guide is written for both developers and customers. No prior experience with pipeline engineering or data science is required.

---

## 📂 Input File Format

Customers will upload a `.csv` (Comma-Separated Values) file containing data about components and anomalies (defects) found along a pipeline.

Each **row** represents a single observation or event—either a pipeline component (e.g., a weld or tee) or an anomaly (e.g., a corrosion site).

---

## 📊 Column Descriptions

### 1. `log dist. [m]`
- **Meaning**: The distance (in meters) from the origin of the pipeline to the location of the observed component or defect.
- **Notes**:
  - Values can be **positive** (downstream) or **negative** (upstream) relative to the pipeline’s origin (defined as `0` meters).
  - Used for plotting defects along the pipeline.

---

### 2. `component / anomaly identification`
- **Meaning**: A label describing what was found at this point on the pipeline.
- **Examples**:
  - `Weld`, `Tee`, `Sleeve begin`, `Sleeve end` (structural components)
  - `Corrosion`, `Corrosion cluster`, `Off take` (defects or anomalies)
- **Note**:
  - This column may include both **pipeline components** and **defects**.
  - Internally, the application may split this into two fields: `component_type` and `anomaly_type`.

---

### 3. `joint number`
- **Meaning**: The identifier of the pipe joint to which this record belongs.
- **Note**: Each entry belongs to exactly one joint.

---

### 4. `joint length [m]`
- **Meaning**: The total length (in meters) of the joint this observation belongs to.
- **Use**: Useful for understanding how defects are distributed across joints.

---

### 5. `wt nom [mm]`
- **Full Name**: Wall Thickness Nominal
- **Meaning**: The expected wall thickness (in millimeters) for the joint or pipe at the time of installation.
- **Use**: Critical for fitness-for-service calculations and risk assessment.

---

### 6. `up weld dist. [m]`
- **Meaning**: Distance (in meters) from the observation point to the nearest **upstream weld**.
- **Notes**:
  - The sign (positive or negative) may vary **between datasets**, but will be **consistent within** each dataset.
  - This can help identify whether defects tend to cluster near welds.

---

### 7. `clock`
- **Meaning**: The approximate **clock position** of the defect around the pipe circumference, based on a clock face analogy.
- **Example**:
  - `6` means bottom-center, `12` means top-center, `3` means right-side, etc.
- **Use**: Helps locate where on the pipe the issue occurred.

---

### 8. `depth [%]`
- **Meaning**: The estimated depth of the anomaly as a **percentage** of the original wall thickness.
- **Example**: A value of `40%` means the defect penetrates 40% into the pipe wall.
- **Use**: Higher values generally indicate more severe damage.

---

### 9. `ERF B31G`
- **Full Name**: Estimated Repair Factor (B31G method)
- **Meaning**: A calculated score estimating defect severity according to the B31G standard.
- **Use**: Lower ERF values usually mean the defect is more severe.

---

### 10. `length [mm]`
- **Meaning**: The physical length (in millimeters) of the defect along the pipe’s axis.
- **Use**: Along with width and depth, gives a full picture of defect geometry.

---

### 11. `width [mm]`
- **Meaning**: The width (in millimeters) of the defect around the pipe’s circumference.

---

### 12. `surface location`
- **Meaning**: Indicates whether the defect is located on the **internal** or **external** surface of the pipe wall.
- **Possible Values**:
  - `INT` → Internal
  - `NON-INT` → External
  - `n/a` → Unknown or not recorded

---

### 13. `comments`
- **Meaning**: Free-text comments or annotations related to the observation.
- **Use**: May contain additional notes from the inspector or data collector.

---

### 14. `ERF RST EA`
- **Meaning**: Estimated Repair Factor calculated using the **RST EA** method.
- **Use**: Another way to assess severity, depending on customer preferences or regulations.

---

### 15. `ERF RST 085`
- **Meaning**: Estimated Repair Factor calculated using the **RST 0.85** method.
- **Use**: Often used for more conservative safety margins.

---

### 16. `ERF RST ASME`
- **Meaning**: Estimated Repair Factor according to **ASME (American Society of Mechanical Engineers)** guidelines.
- **Use**: This is a widely accepted industry standard for safety assessment.

---

## ✅ Summary

| Column Name              | Description                                   |
|--------------------------|-----------------------------------------------|
| `log dist. [m]`          | Distance along pipeline from origin           |
| `component / anomaly`    | What was found at that point                  |
| `joint number`           | Identifier for the pipe joint                 |
| `joint length [m]`       | Length of that joint                          |
| `wt nom [mm]`            | Nominal wall thickness                        |
| `up weld dist. [m]`      | Distance to upstream weld                     |
| `clock`                  | Clock-face position of the defect             |
| `depth [%]`              | Penetration depth of defect                   |
| `ERF B31G`               | Severity by B31G formula                      |
| `length [mm]`            | Length of defect                              |
| `width [mm]`             | Width of defect                               |
| `surface location`       | INT / NON-INT / n/a                           |
| `comments`               | Optional notes or annotations                 |
| `ERF RST EA`             | RST EA-based severity score                   |
| `ERF RST 085`            | RST 0.85-based severity score                 |
| `ERF RST ASME`           | ASME-based severity score                     |

---

For more information or to request features, please open an issue in this repository or contact the support team.

