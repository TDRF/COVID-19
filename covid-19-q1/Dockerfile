FROM python:3

# Do not change
LABEL challenge="covid19"

# Enable training
LABEL enable_training=true

# Description of the model (max 500 characters)
LABEL description="COVID diagnosis prediction"

# Comma-separated list of arbitrary feature names used by the model in desc
# order of importance.
LABEL ranked_features="cough,body temperature,hematocrit,diastolic blood pressure,systolic blood pressure"

# Comma-separated list of references (URL)
LABEL references="https://www.medrxiv.org/content/10.1101/2020.03.19.20039099v1"

# Install dependencies
RUN pip install pandas \
  lightgbm \
  sklearn \
  datetime \
  numpy 

# Move to the app folder
RUN mkdir /app
WORKDIR /app

# Copy our python program for training and inference
COPY src/train.py .
COPY src/test.py .

# Copy Bash scripts expected by the IT infrastructure of the EHR DREAM Challenge
COPY train.sh .
COPY test.sh .


# Add executable permission to Bash scripts
RUN chmod +x train.sh
RUN chmod +x test.sh
