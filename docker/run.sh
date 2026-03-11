docker run -d \
-p 8000:8000 \
-v $(pwd)/data:/app/data \
-v $(pwd)/checkpoints:/app/checkpoints \
-v $(pwd)/log:/app/log \
-v $(pwd)/view:/app/view \
--name crack-train \
crack-train