import asyncio
import os
from typing import List, Optional, Tuple, Union
import wave

from google import genai
from google.genai import types


class Point:
  """Represents a geographical point of interest with associated information."""

  def __init__(
      self,
      name: str,
      lng: Union[float, int],
      lat: Union[float, int],
      visited: bool = False,
      ready: bool = False,
      info: str = "",
      audio_path: str = "",
  ):
    """Initializes a Point object.

    Args:
        name: The name or description of the location.
        lng: The longitude of the point.
        lat: The latitude of the point.
        visited: A boolean indicating if the point has been visited.
        ready: A boolean indicating if the point is ready for interaction.
        audio_path: The file system path to an associated audio file.
        info: General information or notes about the point.
    """
    self.name: str = name
    self.lng: Union[float, int] = lng
    self.lat: Union[float, int] = lat
    self.visited: bool = visited
    self.ready: bool = ready
    self.audio_path: str = audio_path
    self.info: str = info
    
  def to_dict(self):
        return {
            "name": self.name,
            "lng": self.lng,
            "lat": self.lat,
            "visited": self.visited,
            "ready": self.ready,
            "audio_path": self.audio_path,
            "info": self.info,
        }

  def __repr__(self) -> str:
    """Returns a string representation of the Point object."""
    return (
        f"Point(name='{self.name}', "
        f"lng={self.lng}, lat={self.lat}, "
        f"visited={self.visited}, ready={self.ready}, "
        f"audio_path='{self.audio_path}', info='{self.info[:30]}...')"
    )

  """Represents a tour, which is a collection of points and has an ID."""


class Tour:

  def __init__(
      self,
      tour_id: str,
      tour_name: str = "",
      tour_guide_personality: str = "",
      audio_output_dir: str = "audio_files",
      user_preferences: str = "general history and famous landmarks",
      points_list: Optional[List[Point]] = None,
  ):
    """Initializes a Tour object.

    Args:
        tour_id: A unique identifier for the tour.
        tour_name: An optional descriptive name for the tour.
        tour_guide_personality: An optional description of the tour guide's
          personality.
        audio_output_dir: The directory where audio files should be saved.
        user_preferences: User preferences for prompts
        points_list: An optional list of Point objects to initialize the tour

    Args:
        tour_id: A unique identifier for the tour.
        tour_name: An optional descriptive name for the tour.
        tour_guide_personality: An optional description of the tour guide's
          personality.
        points_list: An optional list of Point objects to initialize the tour
          with.
    """
    self.tour_id: str = tour_id
    self.tour_name: str = tour_name if tour_name else tour_id
    self.tour_guide_personality: str = tour_guide_personality
    self.user_preferences: str = user_preferences
    self.points: List[Point] = []

    self.audio_output_dir = audio_output_dir

    if points_list:
      for point in points_list:
        self.add_point(point)

  def to_dict(self):
        return {
            "tour_id": self.tour_id,
            "tour_name": self.tour_name,
            "tour_guide_personality": self.tour_guide_personality,
            "user_preferences": self.user_preferences,
            "audio_output_dir": self.audio_output_dir,
            "points": [point.to_dict() for point in self.points],
        }

  def add_point(self, point: Point) -> None:
    """Adds a Point object to the tour.

    Args:
        point: The Point object to add.
    """
    if isinstance(point, Point):
      self.points.append(point)
    else:
      raise TypeError("Only Point objects can be added to a Tour.")

  def get_point_by_location(self, location_name: str) -> Union[Point, None]:
    """Retrieves a point from the tour by its location name.

    Args:
        location_name: The location name of the point to find.

    Returns:
        The Point object if found, otherwise None.
    """
    for point in self.points:
      if point.name == location_name:  # changed from point.location
        return point
    return None

  def __repr__(self) -> str:
    """Returns a string representation of the Tour object."""
    return (
        f"Tour(tour_id='{self.tour_id}', tour_name='{self.tour_name}',"
        f" tour_guide_personality='{self.tour_guide_personality}',number_of_points={len(self.points)},"
        f" points={self.points}, audio_output_dir='{self.audio_output_dir}')"
    )


async def _generate_content_for_point_async(
    client: genai.client.AsyncClient, tour: Tour, point: Point, index: int
):
  if not client:
    print(
        "GenAI client not initialized. Skipping content generation for"
        f" {point.name}"  # changed from point.location
    )
    point.info = "Skipped: GenAI client not available."
    point.ready = False
    tour.points[index] = point
    return
  total_points = len(tour.points)
  if index == 0:
    tour_position = "start"
  elif index == total_points - 1:
    tour_position = "end"
  else:
    tour_position = "middle"
  previously_visited_places = [
      p.name for i, p in enumerate(tour.points) if i < index  # changed from p.location
  ]
  print(
      f"Async: Generating content for: {point.name} (Position:"
      f" {tour_position})"  # changed from point.location
  )
  try:
    guide_prompt_text = create_tour_guide_prompt(
        location=point.name,  # changed from point.location
        tour_position=tour_position,
        previously_visited_places=previously_visited_places,
        user_preferences=tour.user_preferences,
        tour_guide_personality=tour.tour_guide_personality,
    )
    text_gen_config = types.GenerateContentConfig(
        temperature=1.5,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
    text_model = "models/gemini-2.5-pro-preview-05-06"
    text_response = await client.models.generate_content(
        model=text_model,
        contents=[
            types.Content(
                role="user", parts=[types.Part(text=guide_prompt_text)]
            )
        ],
        config=text_gen_config,
    )
    generated_text = ""
    if text_response.candidates and text_response.candidates[0].content.parts:
      generated_text = text_response.candidates[0].content.parts[0].text
    point.info = generated_text.strip()
    print(
        f"Async: Generated text for {point.name} (first 100 chars):"
        f" {point.info[:100]}..."  # changed from point.location
    )
    if not point.info:
      print(
          f"Async: No text generated for {point.name}. Skipping audio"
          " generation."  # changed from point.location
      )
      print(f"couldn't generate text for point: {point.name}")  # changed from point.location
      tour.points[index] = point
      return
    chosen_voice_name = "Charon"
    print(f"Async: Chosen voice for {point.name}: {chosen_voice_name}")  # changed from point.location
    tts_model = "models/gemini-2.5-flash-preview-tts"
    tts_input_text = (
        "Please narrate the following tour information. Embody a"
        f" {tour.tour_guide_personality} style. Speak clearly and at a"
        f" moderate pace: {point.info}"
    )
    audio_response = await client.models.generate_content(
        model=tts_model,
        contents=[
            types.Content(role="user", parts=[types.Part(text=tts_input_text)])
        ],
        config=types.GenerateContentConfig(
            response_modalities=["AUDIO"],
            speech_config=types.SpeechConfig(
                voice_config=types.VoiceConfig(
                    prebuilt_voice_config=types.PrebuiltVoiceConfig(
                        voice_name=chosen_voice_name,
                    )
                )
            ),
        ),
    )
    audio_data = None
    if audio_response.candidates and audio_response.candidates[0].content.parts:
      audio_part = audio_response.candidates[0].content.parts[0]
      if hasattr(audio_part, "inline_data") and hasattr(
          audio_part.inline_data, "data"
      ):
        audio_data = audio_part.inline_data.data
    if audio_data:
      safe_location_name = "".join(
          c if c.isalnum() else "_" for c in point.name  # changed from point.location
      )
      output_filename = f"{tour.tour_id}_{safe_location_name}_{index}.wav"
      point.audio_path = os.path.join(tour.audio_output_dir, output_filename)
      await asyncio.to_thread(wave_file, point.audio_path, audio_data)
      print(
          f"Async: Generated audio for {point.name} at {point.audio_path}"  # changed from point.location
      )
    else:
      print(f"Async: No audio data generated for {point.name}")  # changed from point.location
    point.ready = True
  except Exception as e:
    print(f"Async: Error generating content for point {point.name}: {e}")  # changed from point.location
    point.info = f"Error generating content: {str(e)}"
    point.audio_path = ""
    point.ready = False
  tour.points[index] = point
  return


async def generate_all_points_content_async(
    client: genai.client.AsyncClient, tour: Tour
):
  if not client:
    print("GenAI client not initialized. Cannot generate tour content.")
    return
  if not tour.points:
    print("No points in the tour to generate content for.")
    return
  print(f"Starting asynchronous content generation for tour: {tour.tour_name}")
  tasks = []
  for i, point in enumerate(tour.points):
    point.ready = False
    point.info = "Processing..."
    point.audio_path = ""
    task = asyncio.create_task(
        _generate_content_for_point_async(client, tour, point, i)
    )
    tasks.append(task)
  await asyncio.gather(*tasks)
  print(
      "Finished all asynchronous content generation tasks for tour:"
      f" {tour.tour_name}"
  )


def create_tour_guide_prompt(
    location: str,
    tour_position: str,
    previously_visited_places: list[str] | None = None,
    user_preferences: str = "general history and famous landmarks",
    tour_guide_personality: str = "friendly and informative",
) -> str:
  """Creates a detailed prompt for an AI tour guide model.

  (Copied and adapted from playground.py - full implementation as in
  playground.py)
  """
  valid_positions = ["start", "middle", "end"]
  if tour_position.lower() not in valid_positions:
    pass

  tour_position_val = tour_position.lower()

  if (
      previously_visited_places
      and isinstance(previously_visited_places, list)
      and len(previously_visited_places) > 0
  ):
    visited_places_str = ", ".join(previously_visited_places)
    visited_places_prompt_section = (
        f"5.  **Previously Visited Places:** {visited_places_str}\n (You may"
        " make brief, relevant connections to these places if it enhances"
        " context for the current location, but do not describe them in"
        f" detail. The focus is **{location}**.)"
    )
  else:
    visited_places_prompt_section = "5.  **Previously Visited Places:** None"

  prompt = f"""
You are an expert tour guide AI. Your task is to generate a transcription of a tour guide's speech for a specific **{location}**.
**IMPORTANT CONTEXT:** The end-user listening to this tour is **physically present** at the **{{location}}**, having arrived based on its geographical coordinates. Your narration must reflect this immediacy and be grounded in observable reality.

This location is part of a larger tour, and its position is **{tour_position_val}**. You will also be provided with a list of places visited before the current one. The transcription must include descriptions of *how* the tour guide delivers their lines, formatted as a descriptive phrase followed by a colon before the dialogue.

**User Inputs:**

1.  **Location:** {location}
2.  **Tour Position:** {tour_position_val} (This segment is the '{tour_position_val}' of the overall tour)
3.  **User Preferences:** {user_preferences}
4.  **Tour Guide Personality:** {tour_guide_personality}
{visited_places_prompt_section}

**Your Task:**
(Task description as in playground.py - e.g., Utilize Google Search, Embody Personality, Contextualize, Craft Transcription)
1.  **Utilize Google Search for Verifiable Information:** You have access to the Google Search tool. Use it extensively to gather accurate, up-to-date, and comprehensive information about the **{location}**.
    * **Focus on:** Permanent features, historical facts, architectural details, cultural significance, generally expected or characteristic experiences, and typical ambiances of the **{location}**.
    * **Avoid:** Inventing transient details. Base your descriptions on what a user physically present can generally observe or learn about.
2.  **Embody the Personality:** Adopt the **{tour_guide_personality}** requested by the user.
3.  **Contextualize Narration Based on Tour Position and Previous Visits:**
    * If **Tour Position is 'start'**: Begin narration as the first major stop.
    * If **Tour Position is 'middle'**: Craft narration transitioning from previous places.
    * If **Tour Position is 'end'**: Deliver narration as the final stop.
4.  **Craft a Tour Guide Transcription with Delivery Cues:** Generate a compelling monologue with delivery cues like (With an inviting gesture): Dialogue.

**Output Format:**
Provide only the transcription of the tour guide's speech.
(Output format description as in playground.py)

**Example Scenario (Illustrating Physical Presence, Position, and Previous Visits):**
(Example as in playground.py)

**Crucial Considerations:**
(Crucial considerations as in playground.py - Factual Realism, Immediacy, etc.)

Now, await the user's input.
"""
  return prompt.strip()


def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
  """Saves PCM audio data to a WAV file."""
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  with wave.open(filename, "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(pcm)