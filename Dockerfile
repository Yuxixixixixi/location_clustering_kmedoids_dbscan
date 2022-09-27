# Use an official Python runtime as a parent image
FROM python:3.8

# Set the working directory to /app
WORKDIR   D:/研究生/BFT算法改进/地理位置聚类/location_clustering-master/location_clustering_kmedoids_dbscan

# Copy the current directory contents into the container at /app
COPY . /location_clustering-master

# Install any needed packages specified in requirements.txt
RUN ["python", "-m", "pip", "install", "--upgrade", "pip"]
COPY requirements.txt /tmp/requirements.txt
RUN ["pip3", "install", "-r", "/tmp/requirements.txt"]

# Make port 80 available to the world outside this container
# EXPOSE 80

# Run app.py when the container launches
CMD ["python", "/location_clustering-master/nodes_location.py"]