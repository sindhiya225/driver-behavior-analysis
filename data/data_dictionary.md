# Driver Behavior Analysis - Data Dictionary

## Dataset Overview
This dataset contains telematics data from vehicle sensors capturing driving behavior patterns. Each row represents aggregated metrics from a single driver/trip.

**Source:** Vehicle sensor data (GPS, accelerometer, OBD-II)
**Collection Method:** Time-series sampling at various percentiles
**Total Records:** Multiple driver sessions
**Time Period:** Continuous monitoring data

## Column Categories

### 1. Time Position (tPos) Metrics
Columns starting with `tPos_` represent time-based position measurements at different percentiles.

#### Positive Time Position (`tPos_+ve_*`)
- **Description:** Time measurements for positive velocity changes/acceleration events
- **Unit:** Seconds or normalized time units
- **Percentiles:** 0.1, 0.2, 0.3, ..., 0.99975 (various thresholds)
- **Example:** `tPos_+ve_0.100000` = 10th percentile of positive time positions

#### Negative Time Position (`tPos_-ve_*`)
- **Description:** Time measurements for negative velocity changes/deceleration events
- **Unit:** Seconds or normalized time units
- **Percentiles:** 0.01, 0.02, ..., 0.9 (various thresholds)
- **Example:** `tPos_d1_-ve_0.010000` = 1st percentile of negative time positions for derivative 1

#### Standard Deviations
- `tPos_+ve_std`: Standard deviation of positive time positions
- `tPos_-ve_std`: Standard deviation of negative time positions
- `tPos_d1_+ve_std`: Standard deviation of positive time position derivatives
- `tPos_d1_-ve_std`: Standard deviation of negative time position derivatives

### 2. Speed Metrics
Columns starting with `speed_` represent velocity measurements.

#### Positive Speed (`speed_+ve_*`)
- **Description:** Speed measurements at various percentiles during acceleration/maintenance
- **Unit:** Miles per hour (mph) or km/h
- **Percentiles:** 0.1, 0.2, ..., 0.99975
- **Example:** `speed_+ve_0.500000` = Median (50th percentile) speed

#### Standard Deviation
- `speed_+ve_std`: Standard deviation of speed measurements

### 3. Acceleration Metrics
Columns starting with `accel_` represent acceleration/deceleration measurements.

#### Positive Acceleration (`accel_+ve_*`)
- **Description:** Acceleration (speed increase) measurements
- **Unit:** m/s²
- **Percentiles:** 0.1, 0.2, ..., 0.99
- **Example:** `accel_+ve_0.900000` = 90th percentile acceleration

#### Negative Acceleration (`accel_-ve_*`)
- **Description:** Deceleration (speed decrease) measurements
- **Unit:** m/s² (negative values)
- **Percentiles:** 0.01, 0.02, ..., 0.9
- **Example:** `accel_-ve_0.050000` = 5th percentile deceleration

#### Standard Deviations
- `accel_+ve_std`: Standard deviation of positive acceleration
- `accel_-ve_std`: Standard deviation of negative acceleration

### 4. RPM (Revolutions Per Minute) Metrics
Columns starting with `rpm_` represent engine RPM measurements.

#### Positive RPM Changes (`rpm_d1_+ve_*`)
- **Description:** Positive changes in RPM (engine speed increase)
- **Unit:** RPM
- **Percentiles:** 0.1, 0.2, ..., 0.99
- **Example:** `rpm_d1_+ve_0.500000` = Median positive RPM change

#### Negative RPM Changes (`rpm_d1_-ve_*`)
- **Description:** Negative changes in RPM (engine speed decrease)
- **Unit:** RPM
- **Percentiles:** 0.01, 0.02, ..., 0.9
- **Example:** `rpm_d1_-ve_0.100000` = 10th percentile negative RPM change

#### Standard Deviations
- `rpm_d1_+ve_std`: Standard deviation of positive RPM changes
- `rpm_d1_-ve_std`: Standard deviation of negative RPM changes

### 5. Derived Features (Created During Analysis)

#### Risk Scores
- `overall_risk_score`: Composite risk score (0-100)
- `acceleration_risk`: Risk from harsh acceleration (0-1)
- `braking_risk`: Risk from harsh braking (0-1)
- `speeding_risk`: Risk from speeding behavior (0-1)

#### Behavioral Metrics
- `harsh_acceleration_count`: Number of harsh acceleration events
- `harsh_braking_count`: Number of harsh braking events
- `speed_variability`: Coefficient of variation for speed
- `rpm_variability`: Standard deviation of RPM
- `fuel_efficiency`: Calculated fuel efficiency score (0-1)

#### Clustering Features
- `cluster`: Assigned cluster number from K-Means
- `cluster_label`: Human-readable cluster description
- `driver_style`: Behavioral classification (Safe, Aggressive, etc.)

#### Composite Metrics
- `safety_score`: Overall safety performance (0-1)
- `aggressive_index`: Aggressiveness measure (0-1)
- `smooth_driving_score`: Smoothness of driving (0-1)
- `time_consistency`: Consistency in time-based metrics

## Data Quality Notes

### Missing Values
- Some extreme percentiles (0.999+) may have fewer data points
- Standard deviation columns derived from corresponding metrics

### Data Transformation
- Original data has been cleaned and normalized
- Negative values in acceleration/deceleration are expected
- Percentile columns represent distribution cutoffs

### Sampling Strategy
- Data collected at various driving conditions
- Percentiles capture behavior across different scenarios
- Standard deviations indicate variability in driving patterns

## Key Relationships

### Safety Indicators
- High `harsh_braking_count` → Increased risk
- High `speed_variability` → Erratic driving
- High `rpm_variability` → Poor gear management

### Efficiency Indicators
- Low `fuel_efficiency` → Poor driving habits
- High `avg_rpm` → Potential fuel waste
- High `accel_aggressiveness` → Increased fuel consumption

### Risk Factors
- `overall_risk_score` > 70: Requires intervention
- `harsh_accel_count` > threshold: Training needed
- `speed_p90` > speed_limit: Speeding behavior

## Usage Guidelines

### For Analysis:
1. Use percentile columns to understand distribution
2. Standard deviation columns indicate consistency
3. Combine metrics for comprehensive assessment

### For Modeling:
1. Normalize features before clustering
2. Consider correlation between acceleration and RPM
3. Use derived features for better interpretability

### For Reporting:
1. Focus on composite scores for stakeholders
2. Use percentiles to show behavior ranges
3. Highlight outliers in standard deviations