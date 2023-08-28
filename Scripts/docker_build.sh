docker build -t my-tf-gpu:latest .
docker rmi $(docker images --filter "dangling=true" -q --no-trunc)