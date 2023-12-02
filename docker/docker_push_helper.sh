az login --scope https://management.core.windows.net//.default
az acr build --file ./docker/DependencyDockerfile --registry M23CSA018 --image base .
az acr build --file ./docker/FinalDockerfile --registry M23CSA018 --image digits .