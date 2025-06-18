!\bin\bash

tag=aicregistry:5000/${USER}:sofa-docker
docker build . -f Dockerfile \
--tag ${tag} --network=host \
--build-arg USER_ID=$(id -u) --build-arg GROUP_ID=$(id -g) --network=host --bui>
docker push ${tag}

# #!/bin/bash
 
# # Create a "tag" or name for the image
# docker_tag=aicregistry:5000/${USER}:latest
 
# # Build the image by calling on your Dockerfile (named Dockerfile in this ins>
# docker build -f Dockerfile --tag ${docker_tag} --network=host --build-arg USE>
 
# docker push ${docker_tag}
 
# requirement.ext 
# pyyaml
# numpy
# scipy
# pygame
