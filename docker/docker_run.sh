# Build the docker file 
docker build -t base:latest -f ./docker/DependencyDockerfile .
docker build -t digits:latest -f ./docker/FinalDockerfile .
docker run digits:latest

# # Create out volume
# docker volume create mltrain
# # Mount our volume to models directory (where train data is stored)
# docker run -d  -p 80:5000 -v mltrain:/digits/models digits:v1 

