"""Demo WebApp using FastAPI."""

import asyncio
import threading
import json
import os
import uuid
from typing import Any, Dict, List, Optional
import shelve
from contextlib import contextmanager
import uuid


from fastapi import FastAPI, Request, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware

from google import genai

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

async def process_tour_background(tour: resources.Tour):
    tour_id = tour.tour_id
    try:
        print(f"Background processing started for tour_id: {tour_id}")
        await resources.generate_all_points_content_async(client, tour)
        with get_db() as db:
            db[tour_id] = tour
        print(f"Background processing finished and tour saved for tour_id: {tour_id}")
    except Exception as e:
        print(f"Error during background tour processing for tour_id {tour_id}: {e}")

def run_tour_background_processing(tour: resources.Tour):
    try:
        asyncio.run(process_tour_background(tour))
    except Exception as e:
        print(f"Exception in run_background_processing for tour {tour.tour_id}: {e}")

@app.post("/tour")
async def post_tour(request: Request, background_tasks: BackgroundTasks):
    try:
        request_body = await request.json()
        tour_name = request_body.get('tour_name', '').replace(" ", "_")
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
        background_tasks.add_task(run_tour_background_processing, tour)
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
    
@app.get("/tour")
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

def run_nugget_background_processing(tour: resources.Tour, point: resources.Point, nugget_id: str):
    try:
        asyncio.run(process_nugget_background(client, tour, point, nugget_id))
    except Exception as e:
        print(f"Exception in run_background_processing for tour {tour.tour_id}: {e}")

async def process_nugget_background(tour: resources.Tour, point: resources.Point, nugget_id: str):
    with get_db() as db:
        try:
            await resources._generate_follow_up(client, tour, point, nugget_id)
        except Exception as e:
            print(f"Exception in process_nugget_background for nugget {nugget_id}: {e}")
        db[tour.tour_id] = tour


@app.post("/tour/{tour_id}/point/{point_name}") # Changed to app.post
async def post_point_nugget(
    tour_id: str,
    point_name: str,
    request: Request,
    background_tasks: BackgroundTasks
):
    """
    Retrieves details for a specific point within a tour,
    accepts a user's question, and initiates background processing.
    """
    with get_db() as db:
        tour : resources.Tour = db.get(tour_id) #
    if not tour:
        raise HTTPException(status_code=404, detail=f"Tour with ID '{tour_id}' not found")

    found_point: resources.Point | None = None
    # Adjust this logic based on your Tour and Point data structure
    if hasattr(tour, 'points') and isinstance(tour.points, list):
        for point in tour.points:
            if hasattr(point, 'name') and point.name == point_name:
                found_point = point
                break
    # Add other options for finding the point if necessary

    if not found_point:
        raise HTTPException(status_code=404, detail=f"Point with name '{point_name}' not found in tour '{tour_id}'")
    
    first_n = found_point.content[0] if len(found_point.content) > 0 else None
    if not first_n or first_n.ready == False:
        return JSONResponse(
            status_code=400,
            content={"message": "Initial point content is not ready yet."},
            headers={'Access-Control-Allow-Origin': '*'},
        )
        
    request_body = await request.json()
    question: str = request_body.get('question', '')
    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    nugget_id = resources.nugget_id(point_name)
    nugget = resources.PointNugget(id=nugget_id, question=question)
    found_point.add_nugget(nugget)

    background_tasks.add_task(
        run_nugget_background_processing,
        tour_id,
        found_point,
        nugget_id
    ) #
    print(f"Enqueued background task for point '{point_name}' in tour '{tour_id}' with question: '{question[:50]}...'")

    # The response can be the point details, or a confirmation message,
    # or a combination. Here, we return the point details along with the question received.
    return JSONResponse(
            status_code=202,
            content={
                "nugget": nugget.to_dict(),
                "nugget_id": nugget_id,
            },
            headers={'Access-Control-Allow-Origin': '*'},
        )

@app.get("/tour/{tour_id}/point/{point_name}/nugget/{nugget_id}")
async def get_nugget_audio(tour_id: str, point_name: str, nugget_id: str):
    with get_db() as db:
        tour_object :resources.Tour = db.get(tour_id)
    if tour_object is None:
        print(f'Tour with id {tour_id} not found.')
        raise HTTPException(status_code=404, detail=f'Tour with id {tour_id} not found.')
    point : resources.Point = None
    for p in tour_object.points:
        if p.name == point_name:
            point = p
            break
    if not point:
        print(f'Point with name {point_name} not found.')
        raise HTTPException(status_code=404, detail=f'Point with id {point_name} not found.')
    ni = point.nugget_index(nugget_id)
    if ni == -1:
        print(f'Nugget with id {nugget_id} not found.')
        raise HTTPException(status_code=404, detail=f'Nugget with id {nugget_id} not found.')
    nugget = point.content[ni]
    return JSONResponse(status_code=200, content={"nugget": nugget.to_dict()},  headers={'Access-Control-Allow-Origin': '*'})

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