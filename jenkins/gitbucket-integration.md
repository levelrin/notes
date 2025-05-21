## Objective

Jenkins runs CI when a pull request is submitted on [GitBucket](https://github.com/gitbucket/gitbucket).

A pull request from a forked repository should be supported.

Jenkins should be able to use Docker.

## Create Docker Network

In this example, we will run Jenkins and GitBucket servers using Docker.

We should create a Docker network for them to communicate with each other.

```sh
docker network create temp
```

## Start GitBucket

Create a volume for keeping the data:
```sh
docker volume create gitbucket
```

Run a container:
```sh
docker run --rm -d --name gitbucket --network temp -p 8080:80 -v gitbucket:/gitbucket gitbucket/gitbucket:4.38.4 sh -c "java -jar /opt/gitbucket.war --port=80"
```

Since Jenkins will assume GitBucket uses port 80, we need to overwrite the startup command for Gitbucket to use port 80.

Access GitBucket via a browser:
http://localhost:8080

Admin Credentials:
 - ID: root
 - Password: root

## Start Jenkins

Run a container:
```sh
docker run --rm -d -u root --name jenkins --network temp -p 8081:8080 -p 50000:50000 -v jenkins_home:/var/jenkins_home -v /var/run/docker.sock:/var/run/docker.sock jenkins/jenkins:lts-jdk17
```

This will automatically create the `jenkins_home` volume.

Since we want to use Docker, we mount the `docker.sock` for the container.

Access Jenkins via a browser:
http://localhost:8081

`ssh` into the container:
```sh
docker container exec -it jenkins /bin/bash
```

In the Jenkins container's terminal, we will obtain the initial admin password and install Docker CLI.

Obtain the initial admin password:
```sh
cat /var/jenkins_home/secrets/initialAdminPassword
```

Install Docker CLI:
```sh
apt update && apt install -y docker.io && rm -rf /var/lib/apt/lists/*
```

## Install Jenkins Plugins

Manage Jenkins -> Plugins -> Available plugins -> Search [Multibranch Scan Webhook Trigger](https://plugins.jenkins.io/multibranch-scan-webhook-trigger/) and install.

Search [Docker Pipeline](https://plugins.jenkins.io/docker-workflow/) and install.

## Create a pipeline on Jenkins

1. New Item -> Multibranch Pipeline -> OK
1. Fill in Display Name
1. Branch Sources -> Add source -> GitHub
1. Credentials -> Add -> Jenkins -> Fill in the username and password of the user that can clone the repo -> Fill in the ID. ex: root-user -> Add
1. Credentials -> select the credentials you just created
1. Repository HTTPS URL -> ex: http://gitbucket/root/demo -> Validate
1. Discover pull requests from forks -> Trust -> Select Collaborators
1. Scan Repository Triggers -> Check `Scan by webhook`
1. Trigger token: demo
1. Save

## Configure Webhook on GitBucket

1. Go to the repo -> Settings -> Service Hooks -> Add webhook
1. Payload URL: ex: http://jenkins:8080/multibranch-webhook-trigger/invoke?token=demo
1. Which events would you like to trigger this webhook? -> Select Pull request, Pull request review comment, and Push
1. Add webhook

## Test

Now, sending a pull request will trigger the build.

We tested a build triggered by a pull request from a forked repository.

Here is the Jenkinsfile we used:
```jenkinsfile
pipeline {
    agent {
        docker { image 'debian:trixie-slim' }
    }
    stages {
        stage('Build') {
            steps {
                echo 'Build Started'
                sleep time: 10, unit: 'SECONDS'
                echo 'Build Finished'
            }
        }
    }
}
```
