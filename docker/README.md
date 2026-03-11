# Run Docker

```bash
docker compose -f docker/compose.yml build
docker compose -f docker/compose.yml up
```



## Apptainer
Local
```bash
docker compose -f docker/compose.yml build

APPTAINER_NOHTTPS=1 apptainer build --fakeroot docker/exports/sim-eval-image.sif docker-daemon://sim-eval-image:latest
tar -cvf ./docker/exports/sim-eval-image.tar sim-eval-image.sif

```

