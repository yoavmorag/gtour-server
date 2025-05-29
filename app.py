"""Demo WebApp using FastAPI."""

import asyncio
import threading
import json
import os
import uuid
from typing import Any, Dict, List, Optional
import shelve
from contextlib import contextmanager

from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from google import  genai

import resources

# You need to import or define these:
# from . import resources
# from .resources import Tour, Point

app = FastAPI()

# Allow CORS for all origins (adjust as needed)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

DB_PATH = "gtour_db"

# Google genai client
client = genai.Client(api_key=os.environ["GOOGLE_API_KEY"]).aio


@contextmanager
def get_db():
    with shelve.open(DB_PATH) as db:
        yield db

async def process_tour_background(tour):
    tour_id = tour.tour_id
    try:
        print(f"Background processing started for tour_id: {tour_id}")
        await resources.generate_all_points_content_async(client, tour)
        with get_db() as db:
            db[tour_id] = tour
        print(f"Background processing finished and tour saved for tour_id: {tour_id}")
    except Exception as e:
        print(f"Error during background tour processing for tour_id {tour_id}: {e}")

def run_background_processing(tour):
    try:
        asyncio.run(process_tour_background(tour))
    except Exception as e:
        print(f"Exception in run_background_processing for tour {tour.tour_id}: {e}")

@app.post("/tour")
async def post_tour(request: Request, background_tasks: BackgroundTasks):
    try:
        request_body = await request.json()
        tour_name = request_body.get('tour_name', '')
        tour_id = f'{tour_name}_{str(uuid.uuid4())}'
        tour_guide_personality = request_body.get('tour_guide_personality', '')
        user_preferences = request_body.get('user_preferences', '')
        points_data = request_body.get('points', [])

        # Convert the points data to resources.Point objects
        points = [resources.Point(**point) for point in points_data]
        audio_output_dir = './data'

        tour = resources.Tour(
            tour_id=tour_id,
            tour_name=tour_name,
            tour_guide_personality=tour_guide_personality,
            user_preferences=user_preferences,
            points_list=points,
            audio_output_dir=audio_output_dir,
        )

        # Start background processing
        background_tasks.add_task(run_background_processing, tour)
        with get_db() as db:
            db[tour_id] = tour

        resp_points = [point.to_dict() for point in tour.points]

        return JSONResponse(
            status_code=202,
            content={
                'message': 'Tour creation initiated. Processing in background.',
                'tour_id': tour_id,
                'tour_name': tour_name,
                'tour_guide_personality': tour_guide_personality,
                'user_preferences': user_preferences,
                'points': resp_points,
                'audio_output_dir': audio_output_dir,
            },
            headers={'Access-Control-Allow-Origin': '*'},
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f'Failed to initiate tour creation: {e}')

@app.get("/tour/{tour_id}")
async def get_tour(tour_id: str):
    print(f'Tour id: {tour_id}')
    with get_db() as db:
        tour_object = db.get(tour_id)
    if tour_object is None:
        print(f'Tour with id {tour_id} not found.')
        raise HTTPException(status_code=404, detail=f'Tour with id {tour_id} not found.')
    tour_data = {
        'tour_id': tour_object.tour_id,
        'tour_name': tour_object.tour_name,
        'tour_guide_personality': tour_object.tour_guide_personality,
        'user_preferences': tour_object.user_preferences,
        'points': [point.to_dict() for point in tour_object.points],
    }
    print(f'Tour data: {tour_data}')
    return JSONResponse(content=tour_data, headers={'Access-Control-Allow-Origin': '*'})

@app.get("/tour")
async def get_all_tours():
    out_data = {}
    with get_db() as db:
        for key in db:
            tour = db[key]
            out_data[tour.tour_id] = tour.to_dict()
    return JSONResponse(content=out_data, headers={'Access-Control-Allow-Origin': '*'})

@app.get("/internal_error")
async def internal_error():
    raise HTTPException(status_code=500, detail='Uncaught exceptions result in 500 Internal Service Error')

@app.get("/error_with_context")
async def error_with_context():
    raise HTTPException(status_code=418, detail='Context displayed to the end user')

# To run: uvicorn app:app --reload