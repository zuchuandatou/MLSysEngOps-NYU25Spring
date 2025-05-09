#!/bin/bash

# Start backend
uvicorn app:app --host 0.0.0.0 --port 8000 --reload &

# Start frontend
cd frontend
npm start -- --host 0.0.0.0