Data Visualizations – Student Dropout Risk Prediction
This document showcases and explains the visualizations created for Review 2 of the project “Predicting Student Dropout Risk Using Data Analytics.” 
Each chart was carefully selected to convey key insights from the dataset, ensuring clarity, storytelling, and interactivity where applicable.

Folder Structure

/visualizations/
│
├── bar_gender_distribution.png
├── pie_dropout_risk.png
├── scatter_absences_vs_avggrade.png
├── box_failures_vs_grade.png
├── interactive_scatter_absences_vs_grade.html
└── interactive_bar_activities_vs_dropout.html

Static Visualizations (Matplotlib & Seaborn)
1. Gender Distribution – Bar Chart
File: bar_gender_distribution.png

Purpose: Displays the gender split of the student dataset.

Insight: Helps check for any gender imbalance in the data that could affect model bias.

2.  Dropout Risk – Pie Chart
File: pie_dropout_risk.png

Purpose: Shows proportion of students classified as high vs low dropout risk.

Insight: Quickly communicates how severe the dropout issue is in this dataset.

3.  Absences vs. Average Grade – Scatter Plot
File: scatter_absences_vs_avggrade.png

Purpose: Maps number of absences to average grades, colored by dropout risk.

Insight: Students with more absences often score lower and are marked high-risk.

4.  Failures vs. Average Grade – Box Plot
File: box_failures_vs_grade.png

Purpose: Visualizes how past failures impact average grades.

Insight: Students with multiple failures generally have lower average grades, aligning with dropout prediction logic.

 Interactive Visualizations (Plotly)
5.  Interactive Scatter: Absences vs. Grade
File: interactive_scatter_absences_vs_grade.html

Features: Tooltips show grade history, failures, and gender.

Purpose: Lets viewers explore the effect of absences on individual students' performance and risk level.

6.  Interactive Bar Chart: Activities vs Dropout Risk
File: interactive_bar_activities_vs_dropout.html

Purpose: Compares dropout risk between students who participate in extracurricular activities vs those who don’t.

Insight: Shows correlation between involvement and lower dropout risk.

Data Storytelling Summary
These visuals support the key findings of our dropout risk model:

Frequent absences and multiple academic failures are strong indicators of risk.

Participation in extracurricular activities may protect against academic decline.

Data is fairly balanced across gender, allowing unbiased modeling.

Each visual adds a layer to the story—helping faculty or analysts identify at-risk students early and act proactively.

How to View
Static visuals: Open .png files in any image viewer.

Interactive visuals: Open .html files in a web browser (Chrome, Firefox, etc.).
