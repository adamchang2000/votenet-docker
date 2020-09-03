use the docker container to start up an ubuntu image with cuda enabled

first cd to this project parent directory.

run:

command: ``docker run -v $PWD/votenet:/usr/src/app/votenet:z -d --rm adamchang2000/votenet-stuff``

then run:

command: ``docker ps``

to get the container id, then run: 

command: ``docker exec -it <container id> /bin/bash``

to enter into container