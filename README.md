# CourtCheck Project

## Overview
CourtCheck is a computer vision project aimed at detecting and tracking tennis court boundaries and ball movements. Utilizing advanced technologies such as YOLOv5, TrackNet, and Detectron2, this project integrates multiple models to deliver precise court detection and ball tracking capabilities.

## Table of Contents
- [Project Background](#project-background)
- [Project Goals](#project-goals)
- [Process](#process)
  - [Data Collection](#data-collection)
  - [Data Annotation](#data-annotation)
  - [Model Selection](#model-selection)
  - [Integration](#integration)
- [Results and Findings](#results-and-findings)
- [Conclusions](#conclusions)
- [How to Run the Project](#how-to-run-the-project)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

## Project Background
The CourtCheck project was initiated to enhance the analysis of tennis matches by accurately detecting court boundaries and tracking ball movements. The project was developed as part of Aggie Sports Analytics and aimed to provide valuable insights for both players and coaches.

## Project Goals
- **Accurate Court Detection**: Identify the boundaries of a tennis court with high precision.
- **Ball Tracking**: Track the movement of the tennis ball throughout the game.
- **Seamless Integration**: Integrate different models to work together seamlessly for a comprehensive analysis.

## Process

### Data Collection
The initial step involved collecting a diverse set of tennis match videos. These videos provided a robust dataset for training and testing the models.

### Data Annotation
Using tools such as OpenCVAT and Roboflow, we annotated over 20,000 images. The annotations included identifying players, the ball, and key points on the court. This step was crucial for training the models to recognize and differentiate between various elements accurately.

### Model Selection
We employed a combination of models for different tasks:
- **YOLOv5**: Used for player detection due to its high accuracy and speed.
- **TrackNet**: Implemented for ball tracking, leveraging its capability to track fast-moving objects.
- **Detectron2**: Utilized for court detection, benefiting from its advanced object detection capabilities.

### Integration
Integrating these models required significant effort to ensure they worked together seamlessly. Using Python and various data structures, we connected the key points and integrated the x and y coordinates from each model's output. This post-processing step was vital for achieving coherent and accurate results.

## Results and Findings
The integrated model demonstrated an accuracy of 95% in detecting court boundaries and tracking ball movements. Below are some key findings:
- **High Precision**: The models were able to accurately detect and track objects under various conditions.
- **Efficiency**: The integration allowed for real-time processing, making the system viable for live analysis.
- **Scalability**: The methodology can be extended to other sports and applications with similar requirements.

### Sample Output
![Sample Output](images/sample_output.png)

## Conclusions
The CourtCheck project successfully achieved its goals, demonstrating the potential of using integrated models for sports analytics. The project highlighted the importance of comprehensive data annotation, model selection, and seamless integration. The findings suggest that this approach can significantly enhance the analysis and understanding of tennis matches.

## How to Run the Project

### Prerequisites
- Python 3.7+
- TensorFlow
- OpenCV
- Google Colab account

### Installation
Clone the repository and install the required dependencies:
```bash
git clone https://github.com/corypham/CourtCheck.git
cd CourtCheck
pip install -r requirements.txt
