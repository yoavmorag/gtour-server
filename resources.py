import asyncio
import os
from typing import List, Optional, Tuple, Union, Dict
import wave

from google import genai
from google.genai import types
import uuid


class PointNugget:
  """Represents a follow-up question and answer for a point of interest."""
  def __init__(self, id: str = "", question: str = "", answer: str = "", audio_path: str = "", ready: bool = False, played: bool = False):
    self.id = id
    self.question = question
    self.answer = answer
    self.audio_path = audio_path
    self.ready = ready
    self.played = played  
  
  def to_dict(self):
    return {
        "id": self.id,
        "answer": self.answer[5:],
        "audio_path": self.audio_path,
        "ready": self.ready,
        "played": self.played,
    }

def nugget_id(point_name:str) -> str:
  return f"{point_name}_nugget_{str(uuid.uuid4())}"
  
class Point:
  """Represents a geographical point of interest with associated information."""

  def __init__(
      self,
      name: str,
      lng: Union[float, int],
      lat: Union[float, int],
      visited: bool = False,
      content: Optional[List[PointNugget]] = [],
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
    self.content: Optional[List[PointNugget]] = content

  def add_nugget(self, pn: PointNugget) -> None:
    if not self.content:
      self.content = []
    self.content.append(pn)

  def nugget_index(self, id: str) -> int:
    for i, nugget in enumerate(self.content or []):
      if nugget.id == id:
        return i
    return -1
    
  def to_dict(self):
        return {
            "name": self.name,
            "lng": self.lng,
            "lat": self.lat,
            "visited": self.visited,
            "content": [nugget.to_dict() for nugget in self.content] if self.content else [],
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

  def get_point_by_name(self, name: str) -> Union[Point, None]:
    """Retrieves a point from the tour by its location name.

    Args:
        location_name: The location name of the point to find.

    Returns:
        The Point object if found, otherwise None.
    """
    for point in self.points:
      if point.name == name: 
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
) -> Tuple[int, Point]:
  nugget = PointNugget(id = nugget_id(point.name))
  
  if not client:
    print(
        "GenAI client not initialized. Skipping content generation for"
        f" {nugget.id}"
    )
    nugget.answer = "error: GenAI client not initialized"
    point.add_nugget(nugget)
    return index, point
  
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
  try:
    guide_prompt_text = create_tour_guide_prompt(
        location=point.name,
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
    generated_text = generated_text.strip()
    if not generated_text:
      print(f"Async: No text generated for {nugget.id}. Skipping audio generation.")
      raise Exception(
          "No text generated error"
      )
    
    nugget.question =guide_prompt_text
    nugget.answer =  generated_text
    print(
        f"Async: Generated text for {point.name} nugget {nugget.id} (first 100 chars):"
        f" {nugget.answer[:100]}..." 
    )

    chosen_voice_name = "Charon"
    print(f"Async: Chosen voice for {nugget.id}: {chosen_voice_name}")
    tts_model = "models/gemini-2.5-flash-preview-tts"
    tts_input_text = (
        "Please narrate the following tour information. Embody a"
        f" {tour.tour_guide_personality} style. Speak clearly and at a"
        f" moderate pace: {nugget.answer}"
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
      output_filename = f"{tour.tour_id}_{nugget.id}.wav"
      nugget.audio_path = os.path.join(tour.audio_output_dir, output_filename)
      await asyncio.to_thread(wave_file, nugget.audio_path, audio_data)
      print(
          f"Async: Generated audio for {nugget.id} at {nugget.audio_path}"  
      )
    else:
      print(f"Async: No audio data generated for {point.name}")
      raise Exception(
          "No audio data generated error"
      )
    nugget.ready = True

  except Exception as e:
    print(f"Async: Error generating content for point {nugget.id}: {e}")
    nugget.answer = f"error: {e}"

  point.add_nugget(nugget)
  return index, point


async def generate_all_points_content_async(
    client: genai.client.AsyncClient, tour: Tour, db
):
    """
    Generates content for all points in a tour asynchronously and updates
    the database after each point's content is generated.
    """
    if not client:
        print("GenAI client not initialized. Cannot generate tour content.")
        return
    if not tour.points:
        print("No points in the tour to generate content for.")
        return

    print(f"Starting asynchronous content generation for tour: {tour.tour_name}")

    # Create a list of asyncio tasks for each point
    tasks = [
        asyncio.create_task(
            _generate_content_for_point_async(client, tour, point, i)
        )
        for i, point in enumerate(tour.points)
    ]

# Process tasks as they are completed
    for i, future in enumerate(asyncio.as_completed(tasks)):
        # Wait for the next task to complete and get its explicit return value
        completed_index, completed_point = await future
        
        # Now, explicitly update the main tour object with the completed data
        tour.points[completed_index] = completed_point
        
        # Update the database with the guaranteed-to-be-current state of the tour object
        db[tour.tour_id] = tour
        print(f"Task {i+1}/{len(tasks)} done. DB updated for tour: {tour.tour_id} with point: '{completed_point.name}'")

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
**IMPORTANT CONTEXT:** The end-user listening to this tour is **physically present** at the **{{location}}**, having arrived based on its geographical coordinates (latitude/longitude). Your narration must reflect this immediacy and be grounded in observable reality.

This location is part of a larger tour, and its position is **{tour_position}**. You will also be provided with a list of places visited before the current one. The transcription must include descriptions of *how* the tour guide delivers their lines, formatted as a descriptive phrase followed by a colon before the dialogue.

**User Inputs:**

1.  **Location:** {location}
2.  **Tour Position:** {tour_position} (This segment is the '{tour_position}' of the overall tour)
3.  **User Preferences:** {user_preferences}
4.  **Tour Guide Personality:** {tour_guide_personality}
{visited_places_prompt_section}

**Your Task:**

1.  **Utilize Google Search for Verifiable Information:** You have access to the Google Search tool. Use it extensively to gather accurate, up-to-date, and comprehensive information about the **{location}**.
    * **Focus on:** Permanent features, historical facts, architectural details, cultural significance, generally expected or characteristic experiences, and typical ambiances of the **{location}**.
    * **Avoid:** Inventing transient details (e.g., specific items at a market stall that might not always be there *today*, specific current weather conditions, temporary non-historical events, or subjective sensory details not universally perceivable). Base your descriptions on what a user physically present can generally observe or learn about.
2.  **Embody the Personality:** Adopt the **{tour_guide_personality}** requested by the user in your tone, language, and style of explanation.
3.  **Contextualize Narration Based on Tour Position and Previous Visits:**
    * If **Tour Position is 'start'**: Begin your narration as if this is the first major stop. Welcome the travelers to this physical spot.
    * If **Tour Position is 'middle'**: Craft your narration as if you have already guided them through **Previously Visited Places** (if provided) and are now arriving here, at **{location}**. Ensure smooth, physically grounded transitions (e.g., "Now that we've arrived from '{previously_visited_places[-1] if previously_visited_places else 'our previous stop'}' and are standing here at **{location}**...").
    * If **Tour Position is 'end'**: Deliver your narration as if this is the final planned stop. Conclude the experience at this physical location.
4.  **Craft a Tour Guide Transcription with Delivery Cues:** Generate a compelling and informative monologue. The transcription should:
    * Provide interesting, verifiable facts, stories, and insights about the **{location}**.
    * Heavily emphasize information that aligns with the user's **{user_preferences}**.
    * Be engaging and use language appropriate for someone physically present (e.g., "As you see before you...", "Look around at...", "The structure you're now standing in front of...").
    * Be well-structured and easy to follow.
    * Highlight unique, observable aspects or lesser-known historical details relevant to the preferences.
    * **Crucially, precede lines of dialogue with a description of the tour guide's delivery (e.g., tone, pace, emotion, implied action), and incase them in parentheses.** For example: "(With an inviting gesture towards the main entrance):" or "(Voice filled with historical reverence):" or "(Pausing to let you take in the view):".

**Output Format:**

Provide only the transcription of the tour guide's speech. The narration must seamlessly reflect its specified **{tour_position}**, the user's physical presence, and may subtly reference **Previously Visited Places** for context if appropriate. Do not include any introductory phrases like "Here is the transcription:" or any other meta-commentary.

**Example Scenario (Illustrating Physical Presence, Position, and Previous Visits):**

* **User Input - Location:** "Shuk HaCarmel (Carmel Market), Tel Aviv"
* **User Input - Tour Position:** "middle"
* **User Input - Previously Visited Places:** ["Rothschild Boulevard", "Neve Tzedek"]
* **User Input - User Preferences:** "vibrant atmosphere, local produce, and street food"
* **User Input - Tour Guide Personality:** "Energetic local foodie"

**Your Expected Output (A snippet):**

Alright, after that lovely stroll through the historic charm of Neve Tzedek, prepare your senses! We're now diving right into the heart of Tel Aviv – Shuk HaCarmel!
Just listen to the sounds, look at the vibrant colors all around you! This market is an explosion of life, isn't it?
You'll typically see an amazing array of fresh fruits and vegetables piled high, like those pyramids of fragrant spices that are a staple here, and the mountains of halva. The vendors are often calling out, adding to the unique energy of this place. etc...

**Crucial Considerations:**

* **Factual and Observable Realism:** The tour MUST describe aspects of the location that are generally true, observable by someone physically present, or historically verifiable. Do NOT invent specific, transient details (e.g., "that specific merchant waving," or "the taste of that particular sample you just tried"). Focus on verifiable characteristics, typical offerings, architecture, history, and general ambiance.
* **Immediacy for Present User:** Constantly remember the user is **physically at the location**. Use language that acknowledges their direct presence and encourages observation of their actual surroundings.
* **Subtle Integration of Previous Places:** References should be brief and for context.
* **Seamless Transitions:** Based on **{tour_position}**.
* **Accuracy:** Via Google Search for verifiable facts.
* **Relevance:** To **{user_preferences}**.
* **Personality Consistency.**

Now, await the user's input.
"""
  return prompt.strip()

async def generate_follow_up(client: genai.client.AsyncClient, tour: Tour, point: Point, nugget_id: str):
  index = point.nugget_index(nugget_id)
  if index == -1:
    raise ValueError(f"Follow-up with id {nugget_id} not found")
  nugget = point.content[index]
  prompt = create_follow_up_prompt(point=point, nugget=nugget, index= index, tour_guide_personality=tour.tour_guide_personality, user_prefrences=tour.user_preferences)
  text_gen_config = types.GenerateContentConfig(
        temperature=1.5,
        tools=[types.Tool(google_search=types.GoogleSearch())],
    )
  text_model = "models/gemini-2.5-flash-preview-04-17-thinking"
  text_response = await client.models.generate_content(
      model=text_model,
      contents=[
          types.Content(
              role="user", parts=[types.Part(text=prompt)]
          )
      ],
      config=text_gen_config,
  )
  answer = ""
  if text_response.candidates and text_response.candidates[0].content.parts:
    answer = text_response.candidates[0].content.parts[0].text
  nugget.answer = answer

  chosen_voice_name = "Charon"
  tts_model = "models/gemini-2.5-flash-preview-tts"
  tts_input_text = (
        "Please narrate answer the following question as a tour guide. Embody a"
        f" {tour.tour_guide_personality} style. Speak clearly and at a"
        f" moderate pace: {answer}"
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
    output_filename = f"{tour.tour_id}_followup_{nugget.id}.wav"
    out_path = os.path.join(tour.audio_output_dir, output_filename)
    await asyncio.to_thread(wave_file, out_path, audio_data)
    nugget.audio_path = out_path
    nugget.ready = True
    print(
        f"Async: Generated audio for {nugget.question[:50]} at {point.audio_path}"
    )
  else:
    print(f"Async: No audio data generated for {nugget.question[:50]}")
  


def create_follow_up_prompt(point: Point, nugget: PointNugget, index: int, tour_guide_personality: str, user_prefrences: str):
  context = f"Initial Tour Prompt: {point.content[0].question if point.content else ''}\n Generated Tour: {point.content[0].answer if point.content else ''}"

  if index > 1:
    context += "Previous Q&A:\n"
  for follow_up in point.content[1:index]:
    context += f"Q: {follow_up.question}\nA: {follow_up.answer}\n"
  return f"""
  You are a helpful tour guide AI with the who embodies the following personality {tour_guide_personality}. Your task is to answer a follow up 
  question your user asked of you to the best of your ability. I'm going to provide you context of a prompt given to a **different**
  tour guide AI and his generated tour response explaining about the following
  point of interest {point.name} which also might include follow-up questions and there answers.
  I will then provide you with a new follow-up question.

  The input format will be:
  Initial tour prompt: some prompt explaining the previous AI Tour guides instructions.
  Generated tour: some tour guide response.
  Optional: Previous Q&A: Q: Some questions and answers

  Given the following point of interest: The specified location the tour is about.
  Please answer the following question: The new question

  Then you should provide the answer.
    
  I would like you to answer it in a similar way
  and manarisim of the previous tour guide. Please make sure you answer correctly. You should provide only the response. You can also link back to previous
  answers, questions or things refrenced in the context if needed in your response. Also, if possible, try to cater your answer to the user prefrences: {user_prefrences}.
  **Important** you have the google search tool enabled which will allow
  you to ground your answer. Use it so that your answers are correct as mistakes are not acceptable. You will be fired if you're wrong and the user will be
  disappointed. On the other hand, if the answer you provide is correct and well thought out you will get a 1 million dollar bonus and the user will be happy!

  I will now provide the input:

  {context}
  Given the following point of interest: {point.name}
  Please answer the following question: {nugget.question}"""

def wave_file(filename, pcm, channels=1, rate=24000, sample_width=2):
  """Saves PCM audio data to a WAV file."""
  os.makedirs(os.path.dirname(filename), exist_ok=True)
  with wave.open(filename, "wb") as wf:
    wf.setnchannels(channels)
    wf.setsampwidth(sample_width)
    wf.setframerate(rate)
    wf.writeframes(pcm)