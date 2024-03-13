#!/bin/sh
#docker buildx create --use

docker buildx build \
--file Dockerfile \
--platform linux/amd64 \
--cache-from type=registry,ref=rwthika/acdc-notebooks:latest \
--tag rwthika/acdc-notebooks:latest \
--output type=docker \
.