# Fine-tune on restaurant-specific data
model.train(
    data='restaurant_queue.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)